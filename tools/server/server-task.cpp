#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    json out;
    if (is_save) {
        out = json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "n_checkpoints", n_checkpoints },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    } else {
        out = json {
            { "id_slot",    id_slot },
            { "filename",   filename },
            { "n_restored", n_tokens },
            { "n_read",     n_bytes },
            { "n_checkpoints", n_checkpoints },
            { "timings", {
                { "restore_ms", t_ms }
            }},
        };
    }
    return out;
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

// ---------------------------------------------------------------------------
// 内存水位 (防 OOM)
// ---------------------------------------------------------------------------
// prompt cache 跑在容器里, 而容器通常没有内存上限 —— 一旦 --cache-ram 配得比
// "宿主/容器实际能给的"还大, 内核 OOM killer 会在缓存到达限额**之前**直接杀掉
// llama-server (2026-10-01 实测: 32768 MiB 限额 + 45 GiB 宿主 + 18 GiB 模型 → 被杀,
// 且 /cache/kv 一直是空的 —— 溢写路径根本没机会跑)。
//
// 这里给 RAM 层加一个动态上限: min(--cache-ram, 可用内存 − 保留量)。
//   LLAMA_CACHE_RAM_RESERVE_MIB  保留量, 默认 4096 MiB; 0 = 关闭水位保护
// 可用内存: cgroup v2/v1 限额优先 (容器设了 --memory 时), 否则 /proc/meminfo 的
// MemAvailable。触发时最老的条目照常**溢写到 L3 磁盘**(不丢数据), 只是更早换层。
//
// 注意这是动态水位: 缓存自身占用会让 available 下降, 所以稳态下缓存约收敛到
// (可用 − 保留)/2 (推导: S + new ≤ (A − S) − R)。保守是刻意的 —— 宁可早换层,
// 也不让内核 OOM 杀进程。
// ---------------------------------------------------------------------------
namespace {

bool pc_read_u64_file(const char * path, uint64_t & out) {
    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }

    char buf[64] = {0};
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (n == 0) {
        return false;
    }

    if (strncmp(buf, "max", 3) == 0) {
        return false; // cgroup v2: 不限
    }

    char * end = nullptr;
    const unsigned long long v = strtoull(buf, &end, 10);
    if (end == buf) {
        return false;
    }

    out = (uint64_t) v;
    return true;
}

// 进程还能安全使用的内存 (字节; 0 = 读不到)
size_t pc_mem_available() {
    uint64_t lim = 0, cur = 0;

    // cgroup v2 (容器设了 --memory 时以它为准)
    if (pc_read_u64_file("/sys/fs/cgroup/memory.max", lim) &&
        pc_read_u64_file("/sys/fs/cgroup/memory.current", cur) && lim > cur) {
        return (size_t) (lim - cur);
    }

    // cgroup v1
    if (pc_read_u64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes", lim) &&
        pc_read_u64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes", cur) &&
        lim > cur && lim < (1ull << 60)) {
        return (size_t) (lim - cur);
    }

    // 容器无内存上限 → 看宿主 (/proc/meminfo 不被 namespace 隔离)
    FILE * f = fopen("/proc/meminfo", "rb");
    if (f == nullptr) {
        return 0;
    }

    char line[256];
    size_t out = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        unsigned long long kb = 0;
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            out = (size_t) kb * 1024;
            break;
        }
    }
    fclose(f);
    return out;
}

size_t pc_ram_reserve() {
    static const size_t v = []() -> size_t {
        const char * s = getenv("LLAMA_CACHE_RAM_RESERVE_MIB");
        if (s != nullptr && *s != '\0') {
            const long mib = strtol(s, nullptr, 10);
            return mib > 0 ? (size_t) mib * 1024 * 1024 : 0; // 0/负 = 关闭水位保护
        }
        return (size_t) 4096 * 1024 * 1024;
    }();
    return v;
}

} // namespace

// limit_size_eff 当前生效的 RAM 上限 = min(--cache-ram, 可用内存 − 保留量)。
// 水位告急(可用 ≤ 保留)时压到 1 字节 → 全部溢写到磁盘 (0 在旧语义里是"不限", 不能用)。
// 水位读不到 / 保护关着时, 退回 --cache-ram 的原始语义。
size_t server_prompt_cache::limit_size_eff() const {
    const size_t reserve = pc_ram_reserve();
    if (reserve == 0) {
        return limit_size;
    }

    const size_t avail = pc_mem_available();
    if (avail == 0) {
        return limit_size; // 读不到水位 → 不额外收紧
    }
    if (avail <= reserve) {
        return 1; // 告急: 全量换层
    }

    const size_t budget = avail - reserve;
    if (limit_size == 0 || budget < limit_size) {
        return budget;
    }
    return limit_size;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // 水位: 上限取 min(--cache-ram, 可用内存 − 保留) —— 见 limit_size_eff()
    const size_t limit_eff = limit_size_eff();

    // skip over-limit entries to avoid disturbing the cache
    if (limit_eff > 0 && state_size_new > limit_eff) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_eff / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_eff > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_eff) {
            SRV_WRN(" - making room for prompt cache entry, pushing oldest entry to disk (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            // L3: 与 update() 一致 —— 磁盘层开着就溢写, 关着则退回旧行为(直接丢)。
            // (旧实现直接 pop_front() 丢掉, 是「溢写」路径的一个漏: 新状态较大时
            //  走的是这条分支, 最老的缓存会被静默丢弃而不是搬到磁盘。)
            disk_write(std::move(states.front()));
            states.pop_front();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    return &states.back();
}

// save_to_disk_only 状态超过 RAM 限额时的兜底: 不塞进 RAM 层, 直接构造一份临时 state 落盘。
// 为什么需要(2026-10-01): 137k token 的会话状态可达 10+ GiB, 一旦超过 --cache-ram (或
// 被水位收紧后的上限), 旧行为是**完全不缓存** → 下次冷预填 20+ 分钟; 而 L3 磁盘装得下。
// 磁盘层关着时返回 false, 与旧行为完全一致。
bool server_prompt_cache::save_to_disk_only(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft,
                                            int32_t id_slot, size_t state_size_tgt, size_t state_size_dft) {
    if (disk_dir.empty()) {
        return false; // 磁盘层没开 → 与旧行为一致: 不缓存
    }

    // 已经被 RAM 层里更长的状态包含 → 不必落盘
    for (const auto & st : states) {
        if (st.prompt.tokens.get_common_prefix(prompt.tokens) == (int) prompt.tokens.size()) {
            return false;
        }
    }

    // 落盘要先把整份状态构到内存里 —— 水位不够就放弃, 避免"为了缓存反而 OOM"
    const size_t need  = state_size_tgt + state_size_dft;
    const size_t avail = pc_mem_available();
    if (avail != 0 && avail < need + pc_ram_reserve()) {
        SRV_WRN(" - prompt state %.3f MiB exceeds RAM limit and memory is tight (available %.3f MiB), not cached\n",
                need / (1024.0 * 1024.0), avail / (1024.0 * 1024.0));
        return false;
    }

    server_prompt_cache_state st;
    st.prompt.tokens      = prompt.tokens.clone();
    st.prompt.checkpoints = prompt.checkpoints;

    try {
        st.data.main.resize(state_size_tgt);
        st.data.drft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for oversized prompt state: %s\n", e.what());
        return false;
    }

    llama_state_seq_get_data_ext(ctx_tgt, st.data.main.data(), state_size_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (ctx_dft != nullptr) {
        llama_state_seq_get_data_ext(ctx_dft, st.data.drft.data(), state_size_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    }

    SRV_WRN(" - prompt state %.3f MiB exceeds RAM limit, spilling straight to disk (L3)\n", need / (1024.0 * 1024.0));

    return disk_write(std::move(st));
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);
    } else {
        // L3 磁盘层(冷缓存): RAM 没命中 → 按 token 前缀从磁盘召回, 判据与上面完全一致
        disk_load(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot, f_keep_best, f_sim_best);
    }

    return true;
}

void server_prompt_cache::update() {
    // 水位保护: 上限可能被「可用内存 − 保留」收紧 (见 limit_size_eff)
    const size_t limit_eff = limit_size_eff();
    if (limit_eff != limit_size && limit_eff != limit_size_wm_last) {
        SRV_WRN(" - RAM watermark: prompt cache limit clamped %.3f MiB -> %.3f MiB (reserve %.3f MiB)\n",
                limit_size / (1024.0 * 1024.0), limit_eff / (1024.0 * 1024.0),
                pc_ram_reserve() / (1024.0 * 1024.0));
        limit_size_wm_last = limit_eff;
    }

    if (limit_eff > 0) {
        while (!states.empty() && size() > limit_eff) {
            SRV_WRN(" - cache size limit reached, pushing oldest entry to disk (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            // L3: 磁盘层开着就溢写到磁盘(后台线程), 关着则退回旧行为(直接丢)
            disk_write(std::move(states.front()));
            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_eff > 0 ? std::max<size_t>(limit_tokens, limit_eff/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, pushing oldest entry to disk (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            // L3: 同上, 溢写到磁盘而不是丢弃
            disk_write(std::move(states.front()));
            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_prompt_cache :: L3 磁盘层 (冷缓存)
//
// 与 RAM 层存的是同一份东西: tokens + context checkpoints + KV 字节 (data.main/drft)。
// 唯一的差别是位置 —— 内存装不下时溢写到磁盘, 召回时读回来, 走的还是同一套
// llama_state_seq_set_data_ext 注入路径。寻址按 token 前缀 (不用会话指纹)。
//
namespace {

constexpr uint32_t KVCD_MAGIC   = 0x4b564344; // 'C''D''V''K' (固定即可, 只求自洽)
constexpr uint32_t KVCD_VERSION = 2; // v2: 头部含模型签名 (v1 文件一律忽略)

// disk_load 等待在飞写入落盘的上限 (毫秒)。GB 级文件写 10-20 秒, 给足余量 ——
// 等十几秒远好过整段冷预填 (145k token ≈ 10.5 分钟)。
// 注意: disk_load 在服务器主循环线程上跑, 等待期间整个批处理会暂停 —— 所以上限
// 要短 (宁可放弃回退冷预填, 也不能长时间卡住其他槽的生成)。
constexpr int64_t disk_wait_ms = 30 * 1000;

template<typename T>
void kvcd_w(std::ostream & os, const T & v) {
    os.write(reinterpret_cast<const char *>(&v), sizeof(T));
}

template<typename T>
bool kvcd_r(std::istream & is, T & v) {
    is.read(reinterpret_cast<char *>(&v), sizeof(T));
    return (bool) is;
}

void kvcd_wbuf(std::ostream & os, const std::vector<uint8_t> & b) {
    kvcd_w(os, (uint32_t) b.size());
    if (!b.empty()) {
        os.write(reinterpret_cast<const char *>(b.data()), b.size());
    }
}

bool kvcd_rbuf(std::istream & is, std::vector<uint8_t> & b) {
    uint32_t n = 0;
    if (!kvcd_r(is, n)) {
        return false;
    }
    b.resize(n);
    if (n > 0) {
        is.read(reinterpret_cast<char *>(b.data()), n);
        if (!is) {
            return false;
        }
    }
    return true;
}

uint64_t kvcd_fnv1a(const void * data, size_t size) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// 读一个磁盘条目。head_only = true 时只读 tokens (扫描索引/前缀匹配用), 不读 KV 大块。
bool kvcd_read_entry(const std::string & path, bool head_only, const std::string & expect_sig, server_tokens & tokens, server_prompt_cache_state * st) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        return false;
    }

    uint32_t magic = 0, version = 0, sig_len = 0, has_mtmd = 0, len = 0;
    if (!kvcd_r(is, magic) || !kvcd_r(is, version) || !kvcd_r(is, sig_len)) {
        return false;
    }
    if (magic != KVCD_MAGIC || version != KVCD_VERSION) {
        return false; // 老版本/异版本: 忽略 (v1 没签名)
    }

    // 模型签名校验: 不匹配 = 别人的 KV (共用目录时会出现) → 不认它, 不动它
    if (sig_len > 0 && sig_len < 4096) {
        std::string sig(sig_len, '\0');
        is.read(sig.data(), (std::streamsize) sig_len);
        if (!is) {
            return false;
        }
        if (sig != expect_sig) {
            return false;
        }
    } else if (!expect_sig.empty()) {
        return false; // 要求有签名, 但这份没有
    }

    if (!kvcd_r(is, has_mtmd) || !kvcd_r(is, len)) {
        return false;
    }

    std::vector<char> packed(len);
    if (len > 0) {
        is.read(packed.data(), len);
        if (!is) {
            return false;
        }
    }

    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
    llama_tokens packed_tok(packed.size() / sizeof(llama_token));
    if (!packed.empty()) {
        memcpy(packed_tok.data(), packed.data(), packed.size());
    }

    try {
        tokens = server_tokens::deserialize(packed_tok, has_mtmd != 0);
    } catch (const std::exception & err) {
        SRV_WRN("%s: deserialize failed for '%s': %s\n", __func__, path.c_str(), err.what());
        return false;
    }

    if (head_only) {
        return true;
    }

    uint32_t n_ckpt = 0;
    if (!kvcd_r(is, n_ckpt)) {
        return false;
    }
    for (uint32_t i = 0; i < n_ckpt; ++i) {
        int64_t n_tokens = 0;
        int32_t id_task = 0, pos_min = 0, pos_max = 0;
        if (!kvcd_r(is, n_tokens) || !kvcd_r(is, id_task) || !kvcd_r(is, pos_min) || !kvcd_r(is, pos_max)) {
            return false;
        }
        common_prompt_checkpoint c;
        c.n_tokens = n_tokens;
        c.id_task  = id_task;
        c.pos_min  = pos_min;
        c.pos_max  = pos_max;
        if (!kvcd_rbuf(is, c.data_tgt) || !kvcd_rbuf(is, c.data_dft) || !kvcd_rbuf(is, c.data_spec)) {
            return false;
        }
        st->prompt.checkpoints.push_back(std::move(c));
    }

    if (!kvcd_rbuf(is, st->data.main) || !kvcd_rbuf(is, st->data.drft)) {
        return false;
    }
    st->prompt.tokens = std::move(tokens);

    return true;
}

} // namespace

void server_prompt_cache::disk_init_from_env() {
    // 先算文件名前缀 (只认自己的条目): fnv1a(签名) 的 16 进制
    if (disk_sig.empty()) {
        disk_prefix = "nosig-";
    } else {
        char hex[32];
        snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) kvcd_fnv1a(disk_sig.data(), disk_sig.size()));
        disk_prefix = std::string(hex) + "-";
    }

    const char * dir = getenv("LLAMA_CACHE_DISK_DIR");
    if (dir == nullptr || dir[0] == '\0') {
        return; // 默认关闭: 行为与打补丁前**完全一致**
    }

    disk_dir = dir;

    if (disk_sig.empty()) {
        SRV_WRN("%s: L3 disk cache has NO model signature — do NOT share disk_dir across models (KV layouts differ)\n", __func__);
    }

    const char * mib = getenv("LLAMA_CACHE_DISK_MIB");
    if (mib != nullptr && mib[0] != '\0') {
        const long long v = atoll(mib);
        limit_disk_size = v > 0 ? (size_t) v * 1024ull * 1024ull : 0;
    }

    SRV_TRC("%s: L3 disk cache: dir = %s, limit = %.3f MiB\n", __func__, disk_dir.c_str(),
            limit_disk_size / (1024.0 * 1024.0));
}

// 调用方必须持有 disk_mu。
void server_prompt_cache::disk_scan() {
    if (disk_scanned) {
        return;
    }
    disk_scanned = true;

    if (disk_dir.empty()) {
        return;
    }

    std::error_code ec;
    if (!std::filesystem::exists(disk_dir, ec)) {
        std::filesystem::create_directories(disk_dir, ec);
        if (ec) {
            SRV_WRN("%s: cannot create disk cache dir '%s': %s\n", __func__, disk_dir.c_str(), ec.message().c_str());
            disk_dir.clear(); // 关掉磁盘层, 别每轮都失败一次
        }
        return;
    }

    for (const auto & de : std::filesystem::directory_iterator(disk_dir, ec)) {
        if (ec) {
            break;
        }
        if (!de.is_regular_file()) {
            continue;
        }

        const std::string name = de.path().filename().string();
        if (name.size() < 4 || name.compare(name.size() - 3, 3, ".kv") != 0) {
            continue;
        }
        // 只认自己签名前缀的条目 —— 别的模型/无签名的文件原地不动 (它们归那边用)
        if (name.compare(0, disk_prefix.size(), disk_prefix) != 0) {
            continue;
        }

        disk_entry e;
        e.path = de.path().string();
        {
            std::error_code ec2;
            e.bytes = (size_t) de.file_size(ec2);
            const auto ft = de.last_write_time(ec2);
            e.mtime_us = (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(ft.time_since_epoch()).count();
        }

        // 只读头部拿 tokens (够前缀匹配), 不读 KV 大块
        server_tokens toks;
        if (!kvcd_read_entry(e.path, /*head_only =*/ true, disk_sig, toks, nullptr)) {
            SRV_WRN("%s: dropping unreadable disk cache entry '%s'\n", __func__, name.c_str());
            std::error_code ec3;
            std::filesystem::remove(e.path, ec3);
            continue;
        }
        e.tokens = std::move(toks);

        disk_size += e.bytes;
        disk_index.push_back(std::move(e));
    }

    SRV_TRC("%s: scanned %zu disk cache entries, %.3f MiB\n", __func__, disk_index.size(), disk_size / (1024.0 * 1024.0));

    disk_evict_to_budget();
}

// 调用方必须持有 disk_mu。
void server_prompt_cache::disk_evict_to_budget() {
    if (limit_disk_size == 0) {
        return;
    }

    while (disk_size > limit_disk_size && !disk_index.empty()) {
        size_t idx = 0;
        for (size_t i = 1; i < disk_index.size(); ++i) {
            if (disk_index[i].mtime_us < disk_index[idx].mtime_us) {
                idx = i;
            }
        }

        SRV_WRN(" - disk cache limit reached, removing oldest entry '%s' (%.3f MiB)\n",
                disk_index[idx].path.c_str(), disk_index[idx].bytes / (1024.0 * 1024.0));

        std::error_code ec;
        std::filesystem::remove(disk_index[idx].path, ec);
        disk_size -= std::min(disk_size, disk_index[idx].bytes);
        disk_index.erase(disk_index.begin() + idx);
    }
}

bool server_prompt_cache::disk_write(server_prompt_cache_state st) {
    if (disk_dir.empty()) {
        return false;
    }

    // 文件名 = tokens 的 FNV-1a (同一段 token 序列 → 同一个文件)。
    // 注意: 多模态 prompt 的 serialize() 会抛异常 —— 那就跳过 (文本模型不受影响)。
    std::vector<char> packed;
    try {
        packed = st.prompt.tokens.serialize();
    } catch (const std::exception & err) {
        SRV_WRN("%s: skipping disk spill (cannot serialize tokens: %s)\n", __func__, err.what());
        return false;
    }

    char hex[32];
    snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) kvcd_fnv1a(packed.data(), packed.size()));
    const std::string path = disk_dir + "/" + disk_prefix + hex + ".kv";

    const size_t bytes = st.size() + packed.size() + 64;

    {
        std::lock_guard<std::mutex> lk(disk_mu);
        disk_scan(); // 惰性扫描 (幂等)

        for (const auto & e : disk_index) {
            if (e.path == path) {
                // 2026-10-03: 只信「文件真的在」的条目 —— 写入线程失败时索引条目会残留,
                // 旧逻辑直接 return true 会把这种幽灵条目当成已在盘上, 之后永远召不回。
                std::error_code ec_ex;
                const bool on_disk = std::filesystem::exists(path, ec_ex);
                if (on_disk) {
                    return true; // 同一份已在盘上
                }
                break; // 幽灵条目: 继续往下走, 重新写一份
            }
        }

        disk_entry e;
        e.path     = path;
        e.tokens   = st.prompt.tokens.clone();
        e.bytes    = bytes;
        e.mtime_us = ggml_time_us();

        disk_size += bytes;
        disk_index.push_back(std::move(e));

        disk_evict_to_budget();
    }

    // 登记「在飞写入」: disk_load 选中本条目时会在 disk_wcv 上等它落盘 (见 disk_load)。
    {
        std::lock_guard<std::mutex> lk(disk_wmu);
        disk_writing.insert(path);
    }

    // GB 级 IO 放后台线程 —— update() 在请求热路径上, 不能让它等盘。
    // 这份 state 已经从 states 里移出来了, 没有别人碰它, 所以可以安全地 move 进线程。
    std::thread([this, st = std::move(st), path, packed = std::move(packed)]() mutable {
        std::error_code ec;
        const std::string tmp = path + ".tmp";
        bool ok = false;
        {
            std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
            if (!os) {
                // 2026-10-03: 旧逻辑这里静默返回 —— 索引里却留着条目, 之后召回应声落空
                SRV_WRN("%s: cannot open '%s' for writing, disk spill dropped\n", "disk_write", tmp.c_str());
            } else {
                kvcd_w(os, KVCD_MAGIC);
                kvcd_w(os, KVCD_VERSION);
                kvcd_w(os, (uint32_t) disk_sig.size());
                if (!disk_sig.empty()) {
                    os.write(disk_sig.data(), (std::streamsize) disk_sig.size());
                }
                kvcd_w(os, (uint32_t) st.prompt.tokens.has_mtmd);
                kvcd_w(os, (uint32_t) packed.size());
                if (!packed.empty()) {
                    os.write(packed.data(), packed.size());
                }

                kvcd_w(os, (uint32_t) st.prompt.checkpoints.size());
                for (const auto & c : st.prompt.checkpoints) {
                    kvcd_w(os, (int64_t) c.n_tokens);
                    kvcd_w(os, (int32_t) c.id_task);
                    kvcd_w(os, (int32_t) c.pos_min);
                    kvcd_w(os, (int32_t) c.pos_max);
                    kvcd_wbuf(os, c.data_tgt);
                    kvcd_wbuf(os, c.data_dft);
                    kvcd_wbuf(os, c.data_spec);
                }

                kvcd_wbuf(os, st.data.main);
                kvcd_wbuf(os, st.data.drft);

                os.flush();
                if (!os) {
                    SRV_WRN("%s: write '%s' failed, disk spill dropped\n", "disk_write", tmp.c_str());
                    std::filesystem::remove(tmp, ec);
                } else {
                    ok = true;
                }
            }
        }

        if (ok) {
            std::filesystem::rename(tmp, path, ec);
            if (ec) {
                SRV_WRN("%s: rename '%s' failed: %s\n", "disk_write", path.c_str(), ec.message().c_str());
                std::filesystem::remove(tmp, ec);
                ok = false;
            }
        }

        if (ok) {
            std::lock_guard<std::mutex> lk(disk_mu);
            for (auto & e : disk_index) {
                if (e.path == path) {
                    e.mtime_us = ggml_time_us();
                    break;
                }
            }
        }

        // 无论成败都摘掉「在飞写入」并唤醒等待者 (见 disk_load) —— 失败时等待方会在
        // 超时/醒来后看到文件不存在, 打 WRN 再放弃 (而不是之前的静默冷预填)。
        {
            std::lock_guard<std::mutex> lk(disk_wmu);
            disk_writing.erase(path);
        }
        disk_wcv.notify_all();
    }).detach();

    return true;
}

bool server_prompt_cache::disk_read(const std::string & path, server_prompt_cache_state & out) {
    server_tokens toks;
    if (!kvcd_read_entry(path, /*head_only =*/ false, disk_sig, toks, &out)) {
        return false;
    }
    return true;
}

bool server_prompt_cache::disk_load(server_prompt & prompt, const server_tokens & tokens_new,
        llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
        float f_keep_best, float f_sim_best) {
    if (disk_dir.empty()) {
        return false;
    }

    // 1) 按 token 前缀挑最匹配的条目 (与 RAM 路径同一套 f_keep/f_sim 判据, 同一套基准)
    std::string best;
    size_t      best_bytes = 0;
    // 诊断 (2026-10-03): 记录最接近的 3 条 —— best.empty() 时打出来, 区分「真没有条目」
    // 还是「有条目但都被判据挡住」(以前两种情况都静默, 事后完全查不出)。
    struct cand_diag { size_t len; int lcp; float f_keep; float f_sim; };
    std::vector<cand_diag> top3;
    {
        std::lock_guard<std::mutex> lk(disk_mu);
        disk_scan();

        for (const auto & e : disk_index) {
            if (e.tokens.empty() || tokens_new.empty()) {
                continue;
            }

            const int lcp = (int) e.tokens.get_common_prefix(tokens_new);

            const float f_keep_cur = float(lcp) / e.tokens.size();
            const float f_sim_cur  = float(lcp) / tokens_new.size();

            // 诊断: 维护 top3 (按 f_sim)
            if (top3.size() < 3 || f_sim_cur > top3.back().f_sim) {
                cand_diag d{ e.tokens.size(), lcp, f_keep_cur, f_sim_cur };
                top3.push_back(d);
                std::sort(top3.begin(), top3.end(), [](const cand_diag & a, const cand_diag & b) {
                    return a.f_sim > b.f_sim;
                });
                if (top3.size() > 3) {
                    top3.pop_back();
                }
            }

            // don't trash large prompts (与 load() 里一致)
            if (f_keep_cur < 0.25f) {
                continue;
            }

            if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
                f_keep_best = f_keep_cur;
                f_sim_best  = f_sim_cur;
                best        = e.path;
                best_bytes  = e.bytes;
            }
        }
    }

    if (best.empty()) {
        // 冷缓存里也没有更合适的 —— 打诊断 (最接近的 3 条), 下次有据可查。
        // 降噪: 完全没相似度 (全新会话, 最接近的 f_sim < 0.05) 只打 TRC;
        // 有相似度却被判据挡住 = 可疑情形 (前缀变了/被挤掉), 打 WRN。
        std::string diag;
        for (const auto & d : top3) {
            char buf[128];
            snprintf(buf, sizeof(buf), "[len=%zu lcp=%d f_keep=%.3f f_sim=%.3f]", d.len, d.lcp, d.f_keep, d.f_sim);
            diag += buf;
        }
        const float top_sim = top3.empty() ? 0.0f : top3.front().f_sim;
        if (top_sim >= 0.05f) {
            SRV_WRN("%s: no disk cache entry matches (tokens_new = %zu, base f_keep = %.3f f_sim = %.3f); closest: %s\n",
                    __func__, tokens_new.size(), f_keep_best, f_sim_best, diag.c_str());
        } else {
            SRV_TRC("%s: no disk cache entry matches (tokens_new = %zu); closest: %s\n",
                    __func__, tokens_new.size(), diag.c_str());
        }
        return false; // 冷缓存里也没有更合适的
    }

    // 内存水位: 读回要先把整份状态构到宿主内存里 (GB 级) —— 装不下就放弃召回,
    // 让这轮走冷预填 (而不是把进程推到 OOM)。
    {
        const size_t avail = pc_mem_available();
        if (avail != 0 && best_bytes + pc_ram_reserve() > avail) {
            SRV_WRN(" - disk cache entry is %.3f MiB but only %.3f MiB available (reserve %.3f MiB), skipping recall\n",
                    best_bytes / (1024.0 * 1024.0), avail / (1024.0 * 1024.0), pc_ram_reserve() / (1024.0 * 1024.0));
            return false;
        }
    }

    // 2) 读回来 (GB 级, 秒级 —— 冷召回本来就要付这个代价)
    //    条目可能刚登记、文件还在后台线程写 (.tmp → rename 之间, GB 级要 10-20 秒) ——
    //    此时直接读会扑空 → 静默冷预填 (2026-10-03 实测: 145k token 冷算 10.5 分钟)。
    //    改为在 disk_wcv 上等写入线程收尾 (上限 disk_wait_ms), 把 10 分钟换成十几秒。
    if (!std::filesystem::exists(best)) {
        const int64_t t_wait0 = ggml_time_ms();
        {
            std::unique_lock<std::mutex> lk(disk_wmu);
            if (disk_writing.count(best) > 0) {
                SRV_WRN(" - disk cache entry is still being written, waiting up to %lld ms: %s\n",
                        (long long) disk_wait_ms, best.c_str());
                disk_wcv.wait_for(lk, std::chrono::milliseconds(disk_wait_ms),
                                  [&]() { return disk_writing.count(best) == 0; });
            }
        }
        std::error_code ec_ex;
        if (!std::filesystem::exists(best, ec_ex)) {
            SRV_WRN("%s: disk cache entry '%s' not available after %.1f s wait, skipping recall\n",
                    __func__, best.c_str(), (ggml_time_ms() - t_wait0) / 1000.0);
            return false;
        }
        SRV_WRN(" - disk cache entry ready after %.1f s wait, reading: %s\n",
                (ggml_time_ms() - t_wait0) / 1000.0, best.c_str());
    }

    server_prompt_cache_state st;
    if (!disk_read(best, st)) {
        SRV_WRN("%s: failed to read disk cache entry '%s'\n", __func__, best.c_str());
        return false;
    }

    SRV_TRC(" - loading prompt from disk cache: %d tokens, %zu checkpoints, %.3f MiB\n",
            st.prompt.n_tokens(), st.prompt.checkpoints.size(), st.data.size() / (1024.0 * 1024.0));

    // 3) 注入槽 —— 与 RAM 路径**完全相同**的两步: set_data_ext 灌 KV, 再把
    //    tokens + checkpoints 一起 move 给 prompt (checkpoints 是复用判定的关键)。
    {
        const size_t size = st.data.main.size();
        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, st.data.main.data(), size, id_slot, 0);
        if (n != size) {
            SRV_ERR("%s: failed to restore disk state with size %zu\n", __func__, size);
            return false;
        }
    }

    if (!st.data.drft.empty()) {
        if (ctx_dft == nullptr) {
            SRV_WRN("%s: disk state has draft data but no draft context\n", __func__);
            return false;
        }
        const size_t size = st.data.drft.size();
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, st.data.drft.data(), size, id_slot, 0);
        if (n != size) {
            SRV_WRN("%s: failed to restore disk draft state with size %zu\n", __func__, size);
            return false;
        }
    }

    prompt = std::move(st.prompt);

    // 刚用过: 刷新 mtime, 让它别被 disk_evict_to_budget 当最旧的清掉
    {
        std::error_code ec;
        std::filesystem::last_write_time(best, std::filesystem::file_time_type::clock::now(), ec);
        std::lock_guard<std::mutex> lk(disk_mu);
        for (auto & e : disk_index) {
            if (e.path == best) {
                e.mtime_us = ggml_time_us();
                break;
            }
        }
    }

    return true;
}
