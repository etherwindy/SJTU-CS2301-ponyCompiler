//===- Dialect.h - Dialect definition for the Pony IR ----------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Pony language.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_PONY_DIALECT_H_
#define MLIR_TUTORIAL_PONY_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "pony/ShapeInferenceInterface.h"

/// Include the auto-generated header file containing the declaration of the pony
/// dialect.
#include "pony/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// pony operations.
#define GET_OP_CLASSES
#include "pony/Ops.h.inc"

#endif // MLIR_TUTORIAL_PONY_DIALECT_H_
