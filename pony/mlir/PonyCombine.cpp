//===- PonyCombine.cpp - Pony High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Pony dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "pony/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>
using namespace mlir;
using namespace pony;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "PonyCombine.inc"
} // namespace

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {

  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // TODO: Optimize the scenario: transpose(transpose(x)) -> x
    // Step 1: Get the input of the current transpose.
    // Hint: For op, there is a function: op.getOperand(), it returns the parameter of a TransposeOp and its type is mlir::Value.

    /* 
     *
     *  Write your code here.
     *
     */
  mlir:Value transposeInput = op.getOperand();


    // Step 2: Check whether the input is defined by another transpose. If not defined, return failure().
    // Hint: For mlir::Value type, there is a function you may use: 
    //       template<typename OpTy> OpTy getDefiningOp () const
 	  //       If this value is the result of an operation of type OpTy, return the operation that defines it

    /* 
     *
     *  Write your code here.
     *  if () return failure();
     *
     */
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();
    if (!transposeInputOp) return failure();

    // step 3: Otherwise, we have a redundant transpose. Use the rewriter to remove redundancy.
    // Hint: For mlir::PatternRewriter, there is a function you may use to remove redundancy: 
    //       void replaceOp (mlir::Operation *op, mlir::ValueRange newValues)
    //       The first argument will be replaced by the second argument.

    /* 
     *
     *  Write your code here.
     *
     */
    rewriter.replaceOp(op, transposeInputOp.getOperand());
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
