//===- MLIRGen.h - MLIR Generation from a Pony AST -------------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the Pony language.
//
//===----------------------------------------------------------------------===//

#ifndef PONY_MLIRGEN_H
#define PONY_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace pony {
class ModuleAST;

/// Emit IR for the given Pony moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
} // namespace pony

#endif // PONY_MLIRGEN_H
