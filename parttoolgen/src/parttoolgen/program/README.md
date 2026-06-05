# Program Backend

This package contains the restricted planner-program and legacy V0 symbolic
affordance helpers used by the compiler backend and historical tests.

Current ToolFrame 6.4 benchmark expansion should start from:

```text
bin/toolframe run-spec
parttoolgen.pipeline.task_spec
parttoolgen.image_grounded
parttoolgen.compiler
parttoolgen.evaluation
```

Do not add new benchmark families here. New image-grounded tasks should define a
`toolframe_task_spec_v1`, task-conditioned tool refs, and an explicit
task-level success checker. The `program` package remains a backend and
compatibility layer for ProgramIR/objective execution.

