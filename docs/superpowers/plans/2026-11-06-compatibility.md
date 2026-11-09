# PSCC-Net compatibility implementation plan

Goal: Apply the six reviewed fixes and portable weight loading for a single CUDA GPU, preserving the network and FP32 training losses.

1. Add CPU regression tests using temporary images, lists and tiny checkpoint tensors. Reproduce validation broadcasting, short-list labels, image formats, output paths and partial checkpoint failures before implementation.
2. Centralize project-relative paths and RGB/mask reading. Preserve white-background RGBA conversion and bicubic training resizing; normalize binary masks and reject incompatible shapes. Filter inference inputs and validate empty splits.
3. Remove the unused NumPy alias. Load tensor dictionaries on CPU with explicit weights_only, accepting DataParallel prefixes. Validate all three legacy checkpoint files before applying any state; stop on incompatible/incomplete resumes. Preserve the existing three-file format.
4. Integrate helpers into train/test. Correct validation dimensions, allow variable-size validation samples, make workers configurable, anchor output paths and preserve best weights separately from resumed weights. Do not enable AMP or change model computation.
5. Run unittest discovery on CPU, parse all Python sources and check git diff whitespace. Review changes and document dependency/setup requirements. No GPU benchmark, actual training or model forward pass.

Final verification: source AST parsing, read-only code review and `git diff --check`.
Regression tests use synthetic tensors and temporary fixtures only. Do not execute
them again: the user requested static verification only during implementation.
