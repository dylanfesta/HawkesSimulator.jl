# Agent Protocol - HawkesSimulator.jl

This is an early-stage Julia package for spiking network simulations.

## Current Scope

- Prefer simple, explicit implementations over broad abstractions. The package
  is still finding its core API.
- Keep README/docs claims aligned with the current focus when editing
  user-facing documentation.

## Julia Style

- Write small functions with one clear responsibility.
- Prefer Julia multiple dispatch over large conditional branches when behavior
  depends on model/input/recorder types.
- Use mutating functions with a `!` suffix and return `nothing` unless there is a
  clear reason to return a value.
- Keep allocations visible and intentional. Reuse provided buffers such as
  `input_alloc`, `utility_alloc`, or recorder storage where that matches the
  local design.
- Keep public names, struct fields, and function signatures boring and explicit;
  avoid clever API layers while the model set is small.
- Favor deterministic behavior in tests and examples. Pass an RNG or seed local
  randomness when random behavior matters.

## Other conventions

When dealing with connectivity and recursion, use a "post <- pre" ordering, meaning that the w_ij element of the weight matrix is the weight from neuron j (pre) to neuron i (post). Functions also follow this same convention, with arguments referring to postsynaptic populations first, then presynaptic populations.

## Testing Expectations

- Use the standard Julia package test flow:

  ```sh
  julia --project=. -e 'using Pkg; Pkg.test()'
  ```

- Every new function should have a corresponding unit test. Small helper
  functions are not exempt.
- When changing an existing function, add or update tests that capture the
  changed behavior directly.
- Organize tests into focused `@testset`s by component or behavior. It is fine
  to split tests into additional files and `include` them from
  `test/runtests.jl` as coverage grows.
- Test numerical behavior with explicit tolerances using `isapprox`
  rather than exact equality when floating-point roundoff is possible.
- Cover edge cases that define the API contract, such as empty populations,
  mismatched dimensions, recorder boundaries, and deterministic random inputs.

## Development Workflow

- Before editing, check the worktree with `git status --short` and avoid
  overwriting unrelated user changes.
- Keep changes scoped to the requested behavior. Do not refactor inactive functions
  or files.
- Run the full package test command before finishing when code changes are made.
  If tests cannot run because dependencies or the environment are unavailable,
  report that clearly.
- For documentation-only edits, a test run is optional; still ensure examples and
  commands are accurate.