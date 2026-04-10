use std::collections::HashSet;
use std::path::Path;
use composable::Composable;
use ort::execution_providers::CPUExecutionProvider;
use ort::session::{Session, SessionInputs, SessionOutputs, builder::GraphOptimizationLevel};
use crate::error::UnexpectedModelSchemaError;

use super::Result;
use super::params::RuntimeParameters;
use super::pipeline::Pipeline;


/// A `Model` can load an ONNX model, and run it using the provided pipeline.
pub struct Model {    
    session: Session,
}


impl Model {    
    pub fn new<P: AsRef<Path>>(model_path: P, params: RuntimeParameters) -> Result<Self> {
        let threads = params.threads();
        // Explicitly register CPU EP without arena to prevent unbounded memory allocation.
        // Default ONNX Runtime behavior enables the arena, which with dynamic-shape models
        // (like GLiNER/DeBERTa) can allocate 10-100x the model size.
        let mut eps: Vec<_> = params.into_execution_providers().collect();
        if eps.is_empty() {
            eps.push(CPUExecutionProvider::default().into());
        }

        let session = Session::builder()?
            .with_intra_threads(threads)?
            .with_execution_providers(eps)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_memory_pattern(false)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
        })
    }

    pub fn new_from_bytes(model_bytes: &[u8], params: RuntimeParameters) -> Result<Self> {
        let threads = params.threads();
        let mut eps: Vec<_> = params.into_execution_providers().collect();
        if eps.is_empty() {
            eps.push(CPUExecutionProvider::default().into());
        }

        let session = Session::builder()?
            .with_intra_threads(threads)?
            .with_execution_providers(eps)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_memory_pattern(false)?
            .commit_from_memory(model_bytes)?;

        Ok(Self {
            session
        })
    }

    /// Perform inferences using the provided pipeline and parameters
    pub fn inference<'a, P: Pipeline<'a>>(&'a self, input: P::Input, pipeline: &P, params: &P::Parameters) -> Result<P::Output> {
        // check schema
        self.check_schema(pipeline)?;
        // pre-process
        let (input, context) = pipeline.pre_processor(params).apply(input)?;
        // inference
        let output = self.run(input)?;                
        // post-process
        let output = pipeline.post_processor(params).apply((output, context))?;        
        // ok
        Ok(output)
    }

    pub fn to_composable<'a, P: Pipeline<'a>>(&'a self, pipeline: &'a P, params: &'a P::Parameters) -> impl Composable<P::Input, P::Output> {
        ComposableModel::new(self, pipeline, params)
    }

    /// Check model schema wrt. pipeline expectations
    fn check_schema<'a, P: Pipeline<'a>>(&'a self, pipeline: &P) -> Result<()> {
        if let Some(expected_inputs) = pipeline.expected_inputs() {
            // inputs should be exactly the same sets
            let actual_inputs: HashSet<_> = self.session.inputs.iter().map(|i| i.name.as_str()).collect();
            if !actual_inputs.eq(expected_inputs) {
                return UnexpectedModelSchemaError::new("input", expected_inputs, &actual_inputs).into_err();
            }
        }
        if let Some(expected_outputs) = pipeline.expected_outputs() {
            // for outputs, we just check that the expected ones are present (but having others is ok)
            let actual_outputs: HashSet<_> = self.session.outputs.iter().map(|i| i.name.as_str()).collect();
            if !actual_outputs.is_superset(&expected_outputs) {
                return UnexpectedModelSchemaError::new("output", expected_outputs, &actual_outputs).into_err();
            }
        }
        Ok(())
    }

    fn run(&self, input: SessionInputs<'_, '_>) -> Result<SessionOutputs<'_, '_>> {
        Ok(self.session.run(input)?)
    }

}


/// References a model, a pipeline and some parameters to implement `Composable`
struct ComposableModel<'a, P: Pipeline<'a>> {
    model: &'a Model,
    pipeline: &'a P,
    params: &'a P::Parameters,
}


impl<'a, P: Pipeline<'a>> ComposableModel<'a, P> {
    pub fn new(model: &'a Model, pipeline: &'a P, params: &'a P::Parameters) -> Self {
        Self { model, pipeline, params }
    }
}


impl<'a, P: Pipeline<'a>> Composable<P::Input, P::Output> for ComposableModel<'a, P> {
    fn apply(&self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, self.pipeline, self.params)
    }
}