//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_PONY_SHAPEINFERENCEINTERFACE_H_
#define MLIR_TUTORIAL_PONY_SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace pony {

/// Include the auto-generated declarations.
#include "pony/ShapeInferenceOpInterfaces.h.inc"

} // namespace pony
} // namespace mlir

#endif // MLIR_TUTORIAL_PONY_SHAPEINFERENCEINTERFACE_H_
