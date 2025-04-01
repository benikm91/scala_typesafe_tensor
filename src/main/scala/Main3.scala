package main3

import util.*

import scala.util.Random
import scala.quoted.*
import annotation.implicitNotFound
import scala.compiletime.ops.int._


trait Named[ColNames <: Tuple]
object Named:
  trait Extract[A]:
    type Out <: Tuple

  object Extract:
    given names[Names <: Tuple, A <: Named[Names]]: Extract[A] with {
      type Out = Names
    }
    given noNames[A]: Extract[A] with {
      type Out = EmptyTuple
    }

trait Axis
trait DataAxis[Size <: Int] extends Axis
trait VectorSpaceAxis[Dim <: Int] extends Axis
trait Grid2D[XDim <: Int, YDim <: Int]
trait NamedVectorSpaceAxis[ColNames <: Tuple] extends VectorSpaceAxis[Length[ColNames]] with Named[ColNames]
trait NamedGrid2D[XDim <: Int, YDim <: Int, X <: String, Y <: String] extends Grid2D[XDim, YDim] with Named[(X, Y)]

case class Tensor[A <: Tuple]()

case class EmptyOrEq[A <: Tuple, B <: Tuple]()
object EmptyOrEq {
  given eq[T <: Tuple]: EmptyOrEq[T, T] = EmptyOrEq[T, T]()
  given bothEmpty: EmptyOrEq[EmptyTuple, EmptyTuple] = EmptyOrEq[EmptyTuple, EmptyTuple]()
  given leftEmpty[T1 <: Tuple]: EmptyOrEq[T1, EmptyTuple] = EmptyOrEq[T1, EmptyTuple]()
  given rightEmpty[T2 <: Tuple]: EmptyOrEq[EmptyTuple, T2] = EmptyOrEq[EmptyTuple, T2]()

}

case class NamesMatch[A, B]()
object NamesMatch:
  given namesMatch[A, B](using 
    n1: Named.Extract[A], 
    n2: Named.Extract[B],
    ev: (EmptyOrEq[n1.Out, n2.Out])
  ): NamesMatch[A, B] = NamesMatch[A, B]()

object VectorSpaceOps:

  extension [
    Ax <: Axis, 
    From <: Int, 
    From1Ax <: VectorSpaceAxis[From],
    From2Ax <: VectorSpaceAxis[From],
  ](
    t1: Tensor[(Ax, From1Ax)]
  )

    def dot[To <: Int, ToAx <: VectorSpaceAxis[To]](
      t2: Tensor[(From2Ax, ToAx)]
    )(using
      ev: NamesMatch[From1Ax, From2Ax]
    ): Tensor[(Ax, ToAx)] = 
      new Tensor[(Ax, ToAx)]()


object Grid2DOps:

  trait Convolution[I <: Grid2D[?, ?], K <: Grid2D[?, ?], Stide <: Int]:
    type Out <: Grid2D[?, ?]

  object Convolution:
    type Aux[I <: Grid2D[?, ?], K <: Grid2D[?, ?], Stride <: Int, O <: Grid2D[?, ?]] = Convolution[I, K, Stride] { type Out = O }

    given unnamedGridConv[IX <: Int, IY <: Int, KX <: Int, KY <: Int, Stride <: Int]: Convolution.Aux[Grid2D[IX, IY], Grid2D[KX, KY], Stride, Grid2D[IX - KX + 1, IY - KY + 1]] =
      new Convolution[Grid2D[IX, IY], Grid2D[KX, KY], Stride]:
        type Out = Grid2D[IX - KX + 1, IY - KY + 1]

    given namedGridConv[IX <: Int, IY <: Int, NX <: String, NY <: String, KX <: Int, KY <: Int, Kernel <: Grid2D[KX, KY], Stride <: Int]: Convolution.Aux[NamedGrid2D[IX, IY, NX, NY], Kernel, Stride, NamedGrid2D[IX - KX + 1, IY - KY + 1, NX, NY]] =
      new Convolution[NamedGrid2D[IX, IY, NX, NY], Kernel, Stride]:
        type Out = NamedGrid2D[IX - KX + 1, IY - KY + 1, NX, NY]

  extension [
    Ax <: Axis, 
    InputAx <: Grid2D[?,?],
  ](
    t1: Tensor[(Ax, InputAx)]
  )
    def conv[
      KernelAx <: Grid2D[?,?]
    ](
      kernel: Tensor[Tuple1[KernelAx]],
      stride: Int = 1
    )(using
      ev: NamesMatch[InputAx, KernelAx],
      ot: Convolution[InputAx, KernelAx, stride.type],
    ): Tensor[(Ax, ot.Out)] = 
      new Tensor[(Ax, ot.Out)]()


@main def hello3(): Unit = 
  {
    import VectorSpaceOps.*
    val data = Tensor[(DataAxis[256], VectorSpaceAxis[3])]()
    val linearProjection = Tensor[(VectorSpaceAxis[3], VectorSpaceAxis[10])]()
    val output = data.dot(linearProjection)
    val data2 = Tensor[(DataAxis[256], NamedVectorSpaceAxis[("length", "width", "height")])]()
    val linearProjection2 = Tensor[(NamedVectorSpaceAxis[("length", "width", "height")], VectorSpaceAxis[10])]()
    // val linearProjection2 = Tensor[(NamedVectorSpaceAxis[("width", "length", "height")], VectorSpaceAxis[10])]()
    val output2 = data2.dot(linearProjection2)
  }

  {
    import Grid2DOps.*
    val imgData = Tensor[(DataAxis[256], Grid2D[28, 28])]()
    val kernel = Tensor[Tuple1[Grid2D[3, 3]]]()
    val output = imgData.conv(kernel)
    val imgData2 = Tensor[(DataAxis[256], NamedGrid2D[28, 28, "width", "height"])]()
    imgData2.conv(kernel)
    // val kernel2 = Tensor[Tuple1[NamedGrid2D[3, 3, "height", "width"]]]()  // TODO this should not work!!
    val kernel2 = Tensor[Tuple1[NamedGrid2D[3, 3, "width", "height"]]]()  // TODO this should not work!!
    val output2 = imgData2.conv(kernel2)
  }