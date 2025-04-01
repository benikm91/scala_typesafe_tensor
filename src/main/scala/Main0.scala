/*

A tensor is a array of data [512, Values], where Values can be a Vector space, a 2D-Grid, 3D-Grid, etc.
Operations are defined on Value types. E.g. A Vector space transformation, a 2D-Kernel.

*/

package main0

import util.*

import scala.util.Random
import scala.quoted.*
import scala.Tuple.{Concat, Head, Init, Last, Tail, Size}


def sumTensorAlongAxis[T <: Dims, Ax <: Int](t: Tensor[T], axis: Ax)(
  using ev: LTE[axis.type, Size[T]]
): Tensor[RemoveIndex[T, axis.type]] = Tensor[RemoveIndex[T, axis.type]]()

class Tensor[T <: Dims]:

  def +(that: Tensor[T]): Tensor[T] = 
    new Tensor[T]()

  def -(that: Tensor[T]): Tensor[T] = 
    new Tensor[T]()

  def sum[Ax <: Int](axis: Ax)(
    using ev: LTE[axis.type, Size[T]]
  ) = sumTensorAlongAxis(this, axis)

  def dot[U <: Dims](that: Tensor[U])(
    using ev2: (Last[T] =:= Head[U])
  ): Tensor[Concat[Init[T], Tail[U]]] = 
    new Tensor[Concat[Init[T], Tail[U]]]()

  inline def shape: T = tupleShape[T]

  def swap[Ax1 <: Int, Ax2 <: Int](a1: Ax1, a2: Ax2)(
    using 
    LTE[a1.type, Size[T]],
    LTE[a2.type, Size[T]],
  ): Tensor[Swap[T, a1.type, a2.type]] = Tensor[Swap[T, a1.type, a2.type]]()

object MatrixExtensions:

  type Matrix = Tuple2[Int, Int]

  extension [T <: Matrix](tensor: Tensor[T])
    def transpose: Tensor[Last[T] *: Head[T] *: EmptyTuple] =
      new Tensor[Last[T] *: Head[T] *: EmptyTuple]()

@main def hello(): Unit =

  /** -- BASIC EXAMPLE -- */ 
  {
    println("BASIC EXAMPLE")
    println((Tensor[(2,2)]() + Tensor[(2,2)]()).shape)
  }

  /** -- RUNTIME EXAMPLE -- */ 
  {
    println("RUNTIME EXAMPLE")
    val i1 = new Random().between(1, 10)
    println((Tensor[(i1.type, 2)]() + Tensor[(i1.type, 2)]()).shape)
  }

  /** -- INNER PRODUCT EXAMPLE -- */ 
  {
    println("INNER PRODUCT")
    val t1 = Tensor[(2, 2, 3)]()
    val t2 = Tensor[(3, 2, 2)]()
    val x = t1.dot(t2)
    println(x.shape)
  }

  /** -- NAMED TENSORS EXAMPLE -- */ 
  {
    println("NAMED TENSORS")
    val t1 = Tensor[(("BS", 2), ("I", 3))]()
    val t2 = Tensor[(("I", 3), ("H", 2))]()
    val x: Tensor[(("BS", 2), ("H", 2))] = t1.dot(t2)
    println(x.shape)
  }

  /** -- DATAFRAME EXAMPLE -- */ 
  {
    println("DATAFRAME EXAMPLE")
    println((Tensor[(512, "length")]).shape)
    println((Tensor[(512, ("length", "width", ("name", ("Salmon", "Roach"))))]).shape)
  }

  /** -- TRANSPOSE EXAMPLE -- */ 
  {
    println("TRANSPOSE EXAMPLE")
    import MatrixExtensions.*
    val tt = Tensor[(2, 3)]().transpose
    println(tt.shape)
  }

  /** -- SUM EXAMPLE -- */ 
  {
    def sumExample[BS <: Int, R <: Int, C <: Int](
      x: Tensor[(BS, R, C)],
    ): Tensor[Tuple1[BS]] =
      val h = x.sum(1)
      val r = h.sum(1)
      r
    println("SUM EXAMPLE")
    println(sumExample(Tensor[(512, 128, 128)]()).shape)
    println(sumExample(Tensor[(32, 64, 64)]()).shape)
  }

  /** -- SWAP EXAMPLE -- */ 
  {
    println("SWAP EXAMPLE")
    println(Tensor[(512, 128, 128)]().swap(0, 1).shape)
  }

  /** -- MNIST EXAMPLE -- */
  { 
    def nnExample[BS <: Int](
      x: Tensor[(BS, 784)],
      w1: Tensor[(784, 128)],
      w2: Tensor[(128, 10)],
    ): Tensor[(BS, 10)] =
      val h = x.dot(w1)
      val r = h.dot(w2)
      r

    println("MNIST EXAMPLE")
    println(nnExample(
      Tensor[(512, 784)](),
      Tensor[(784, 128)](),
      Tensor[(128, 10)](),
    ).shape)
  }
