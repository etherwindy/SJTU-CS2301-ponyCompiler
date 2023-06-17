//===- Passes.h - Pony Passes Definition -----------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Pony.
//
//===----------------------------------------------------------------------===//

#ifndef PONY_PASSES_H
#define PONY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace pony {
std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Pony IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

/// Create a pass for lowering operations the remaining `Pony` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace pony
} // namespace mlir

#endif // PONY_PASSES_H
