//! `flare run --max-context-tokens` and `--chat-template`: serving-time replacements for the
//! package's per-request context and Jinja chat template, applied to the package's server
//! configuration before the server validates it. The package file is never written.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Lower the per-request context to `tokens`. The package's KV capacity is already sized for its
/// prepared context, so only a value at or below it is accepted.
pub fn apply_max_context_tokens(config: &mut Value, tokens: usize) -> Result<(), String> {
    let prepared = config["max_sequence_tokens"]
        .as_u64()
        .ok_or("the package declares no prepared context")?;
    if tokens as u64 > prepared {
        return Err(format!(
            "--max-context-tokens {tokens} exceeds the package's context {prepared}"
        ));
    }
    if let Some(page) = config["model_contract"]["cache"]["page_tokens"].as_u64() {
        if page != 0 && tokens as u64 % page != 0 {
            return Err(format!(
                "--max-context-tokens {tokens} must be a multiple of the package's page size {page}"
            ));
        }
    }
    config["max_sequence_tokens"] = tokens.into();
    Ok(())
}

/// A chat template read before the package loads, so a missing file costs no load time.
pub struct ChatTemplate {
    path: std::path::PathBuf,
    sha256: String,
}

impl ChatTemplate {
    pub fn read(path: &Path) -> Result<Self, String> {
        let path = path
            .canonicalize()
            .map_err(|error| format!("--chat-template: cannot access {}: {error}", path.display()))?;
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("--chat-template: cannot read {}: {error}", path.display()))?;
        Ok(Self {
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            path,
        })
    }

    /// Replace the package's `chat_template` asset. A package whose frontend renders no Jinja
    /// template carries no such asset, and a template would never be read.
    pub fn apply(self, config: &mut Value) -> Result<(), String> {
        let Some(assets) = config["assets"].as_object_mut() else {
            return Err("the package's server configuration lists no assets".into());
        };
        if !assets.contains_key("chat_template") {
            return Err("--chat-template: this package's frontend renders no Jinja template".into());
        }
        assets.insert(
            "chat_template".into(),
            json!({"path": self.path, "sha256": self.sha256}),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package_config() -> Value {
        json!({
            "max_sequence_tokens": 262144,
            "model_contract": {"cache": {"page_tokens": 64}},
            "assets": {
                "tokenizer": {"path": "/proc/self/fd/3", "offset": 0, "nbytes": 10, "sha256": "aa"},
                "chat_template": {"path": "/proc/self/fd/3", "offset": 10, "nbytes": 5, "sha256": "bb"},
            },
        })
    }

    #[test]
    fn max_context_equal_to_prepared_is_kept() {
        let mut config = package_config();
        apply_max_context_tokens(&mut config, 262144).unwrap();
        assert_eq!(config["max_sequence_tokens"], json!(262144));
    }

    #[test]
    fn max_context_lower_than_prepared_lowers_it() {
        let mut config = package_config();
        apply_max_context_tokens(&mut config, 131072).unwrap();
        assert_eq!(config["max_sequence_tokens"], json!(131072));
    }

    #[test]
    fn max_context_higher_than_prepared_names_both() {
        let mut config = package_config();
        let error = apply_max_context_tokens(&mut config, 524288).unwrap_err();
        assert_eq!(
            error,
            "--max-context-tokens 524288 exceeds the package's context 262144"
        );
        assert_eq!(config["max_sequence_tokens"], json!(262144));
    }

    #[test]
    fn max_context_off_page_boundary_is_refused() {
        let mut config = package_config();
        let error = apply_max_context_tokens(&mut config, 100000).unwrap_err();
        assert!(error.contains("multiple of the package's page size 64"), "{error}");
    }

    #[test]
    fn chat_template_replaces_the_package_asset() {
        let dir = std::env::temp_dir().join(format!("flare-chat-template-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("template.jinja");
        std::fs::write(&file, "{{ messages }}").unwrap();
        let mut config = package_config();
        ChatTemplate::read(&file).unwrap().apply(&mut config).unwrap();
        let asset = &config["assets"]["chat_template"];
        assert_eq!(asset["path"], json!(file.canonicalize().unwrap()));
        assert_eq!(asset["sha256"], json!(format!("{:x}", Sha256::digest(b"{{ messages }}"))));
        assert!(asset.get("offset").is_none() && asset.get("nbytes").is_none());
        assert_eq!(config["assets"]["tokenizer"], package_config()["assets"]["tokenizer"]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn chat_template_on_a_package_without_jinja_is_refused() {
        let file = std::env::temp_dir().join(format!("flare-chat-template-nj-{}", std::process::id()));
        std::fs::write(&file, "x").unwrap();
        let mut config = package_config();
        config["assets"].as_object_mut().unwrap().remove("chat_template");
        let error = ChatTemplate::read(&file).unwrap().apply(&mut config).unwrap_err();
        assert_eq!(error, "--chat-template: this package's frontend renders no Jinja template");
        std::fs::remove_file(file).unwrap();
    }

    #[test]
    fn chat_template_missing_file_is_refused_before_load() {
        let error = ChatTemplate::read(Path::new("/nonexistent/template.jinja"))
            .err()
            .unwrap();
        assert!(error.starts_with("--chat-template: cannot access /nonexistent/template.jinja"), "{error}");
    }

    #[test]
    fn max_context_and_chat_template_flags_parse_on_run() {
        use clap::Parser;
        let crate::cli::Command::Run(run) = crate::cli::Cli::try_parse_from([
            "flare",
            "run",
            "model.safetensors",
            "--max-context-tokens",
            "131072",
            "--chat-template",
            "/templates/t.jinja",
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        assert_eq!(run.max_context_tokens.map(|n| n.get()), Some(131072));
        assert_eq!(run.chat_template.as_deref(), Some(Path::new("/templates/t.jinja")));
    }
}
