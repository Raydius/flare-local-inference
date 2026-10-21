//! `flare run --generation-defaults`: server-side defaults for requests that omit them,
//! merged into the package's server configuration before the server validates it.

use serde_json::{Map, Value};

const SAMPLING_KEYS: [&str; 8] = [
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
];

/// The parsed `--generation-defaults` object; its keys are known and its values typed.
pub struct GenerationDefaults(Map<String, Value>);

impl GenerationDefaults {
    /// Parse the flag's JSON before the package loads, so a usage error costs no load time.
    pub fn parse(text: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(text)
            .map_err(|error| format!("--generation-defaults is not valid JSON: {error}"))?;
        let Value::Object(object) = value else {
            return Err("--generation-defaults must be a JSON object".into());
        };
        for (key, value) in &object {
            let valid = match key.as_str() {
                "seed" | "top_k" | "max_new_tokens" => value.is_u64() || value.is_i64(),
                "chat_template_kwargs" => value.is_object(),
                key if SAMPLING_KEYS.contains(&key) => value.is_number(),
                _ => {
                    return Err(format!(
                        "--generation-defaults: unknown key {key}; expected {}, max_new_tokens or chat_template_kwargs",
                        SAMPLING_KEYS.join(", ")
                    ));
                }
            };
            if !valid {
                return Err(format!(
                    "--generation-defaults: {key} has the wrong type ({value})"
                ));
            }
        }
        Ok(Self(object))
    }

    /// Merge into the package's server configuration. A sampling default outside the package's
    /// declared range is refused here with the key, value and range; the server's own validation
    /// still runs afterwards.
    pub fn apply(self, config: &mut Value) -> Result<(), String> {
        for (key, value) in self.0 {
            match key.as_str() {
                "max_new_tokens" => config["max_new_tokens"] = value,
                "chat_template_kwargs" => {
                    let defaults = &mut config["template_defaults"];
                    if defaults.is_null() {
                        *defaults = Value::Object(Map::new());
                    }
                    let Some(defaults) = defaults.as_object_mut() else {
                        return Err("the package's template_defaults is not an object".into());
                    };
                    defaults.extend(value.as_object().cloned().unwrap_or_default());
                }
                _ => {
                    let range = &config["sampling_defaults"]["ranges"][&key];
                    if let (Some(min), Some(max), Some(number)) =
                        (range["min"].as_f64(), range["max"].as_f64(), value.as_f64())
                    {
                        if !(min <= number && number <= max) {
                            return Err(format!(
                                "--generation-defaults: {key}={value} is outside the package's range [{}, {}]",
                                range["min"], range["max"]
                            ));
                        }
                    }
                    config["sampling_defaults"][&key] = value;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn package_config() -> Value {
        json!({
            "max_new_tokens": 32768,
            "sampling_defaults": {"seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
                                  "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
                                  "frequency_penalty": 0.0,
                                  "ranges": {"temperature": {"min": 0.6, "max": 1.0}}},
        })
    }

    fn apply(text: &str) -> Result<Value, String> {
        let mut config = package_config();
        GenerationDefaults::parse(text)?.apply(&mut config)?;
        Ok(config)
    }

    #[test]
    fn generation_defaults_merge_sampling_keys() {
        let config = apply(r#"{"temperature": 0.8, "top_k": 40, "seed": 7}"#).unwrap();
        assert_eq!(config["sampling_defaults"]["temperature"], json!(0.8));
        assert_eq!(config["sampling_defaults"]["top_k"], json!(40));
        assert_eq!(config["sampling_defaults"]["seed"], json!(7));
        assert_eq!(config["sampling_defaults"]["top_p"], json!(0.95));
        assert_eq!(config["sampling_defaults"]["ranges"], package_config()["sampling_defaults"]["ranges"]);
    }

    #[test]
    fn generation_defaults_set_max_new_tokens() {
        let config = apply(r#"{"max_new_tokens": 4096}"#).unwrap();
        assert_eq!(config["max_new_tokens"], json!(4096));
    }

    #[test]
    fn generation_defaults_merge_chat_template_kwargs() {
        let config = apply(r#"{"chat_template_kwargs": {"enable_thinking": false}}"#).unwrap();
        assert_eq!(config["template_defaults"], json!({"enable_thinking": false}));
        let mut existing = config.clone();
        existing["template_defaults"] = json!({"add_generation_prompt": true});
        GenerationDefaults::parse(r#"{"chat_template_kwargs": {"enable_thinking": true}}"#)
            .unwrap()
            .apply(&mut existing)
            .unwrap();
        assert_eq!(
            existing["template_defaults"],
            json!({"add_generation_prompt": true, "enable_thinking": true})
        );
    }

    #[test]
    fn generation_defaults_reject_unknown_keys_and_non_objects() {
        let error = apply(r#"{"temprature": 0.8}"#).unwrap_err();
        assert!(error.contains("unknown key temprature"), "{error}");
        assert!(apply("[1]").unwrap_err().contains("must be a JSON object"));
        assert!(apply("{").unwrap_err().contains("not valid JSON"));
        assert!(apply(r#"{"top_k": 0.5}"#).unwrap_err().contains("top_k has the wrong type"));
        assert!(apply(r#"{"chat_template_kwargs": 1}"#)
            .unwrap_err()
            .contains("chat_template_kwargs has the wrong type"));
    }

    #[test]
    fn generation_defaults_out_of_range_names_key_value_and_range() {
        let error = apply(r#"{"temperature": 0.2}"#).unwrap_err();
        assert_eq!(
            error,
            "--generation-defaults: temperature=0.2 is outside the package's range [0.6, 1.0]"
        );
    }

    #[test]
    fn generation_defaults_flag_parses_on_run() {
        use clap::Parser;
        let crate::cli::Command::Run(run) = crate::cli::Cli::try_parse_from([
            "flare",
            "run",
            "model.safetensors",
            "--generation-defaults",
            r#"{"temperature":0.8}"#,
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        assert_eq!(run.generation_defaults.as_deref(), Some(r#"{"temperature":0.8}"#));
    }
}
