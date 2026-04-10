use std::{collections::HashSet, error, fmt::Display};

#[derive(Debug, Clone)]
/// Defines an error caused by a mismatch between pipeline's expected input 
/// or outputs (if any), and the ones of the provided model.
pub struct UnexpectedModelSchemaError {
    message: String,
}

impl UnexpectedModelSchemaError {
    pub fn new(kind: &str, expected: &HashSet<&str>, actual: &HashSet<&str>) -> Self {
        Self {
            message: format!("{} tensors mismatch: pipeline expects {:?} but model has {:?}", kind, actual, expected),
        }
    }

    pub fn with(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }

    pub fn into_err<T>(self) -> super::Result<T> {
        Err(Box::new(self))
    }
}

impl error::Error for UnexpectedModelSchemaError { }

impl Display for UnexpectedModelSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}