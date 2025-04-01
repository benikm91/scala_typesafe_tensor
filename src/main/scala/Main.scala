package main

import scala.compiletime.ops.int._
import java.util.jar.Attributes.Name
import scala.compiletime.{erasedValue, constValue, constValueOpt}
import scala.util.Random
import scala.annotation.targetName
import torch.Float32
import scala.annotation.implicitNotFound
import spire.random.Size
import torch.nn.functional.conv2d
import algebra.lattice.Bool

type __ = "_"  // Empty name for name structure

@implicitNotFound(msg="Are not ${This} or equal: ${A}, ${B}")
case class ThisOrEq[This, A, B]()
object ThisOrEq {
  given eq[This, T1, T2](using T1 =:= T2): ThisOrEq[This, T1, T2] = ThisOrEq[This, T1, T2]()
  given bothEmpty[This]: ThisOrEq[This, This, This] = ThisOrEq[This, This, This]()
  given leftEmpty[T1, This]: ThisOrEq[This, T1, This] = ThisOrEq[This, T1, This]()
  given rightEmpty[This, T2]: ThisOrEq[This, This, T2] = ThisOrEq[This, This, T2]()

}

/** Transforms a tuple `(T1, ..., Tn)` into `(Ti+1, ..., Tn)`. */
type DropLast[T <: Tuple, N <: Int] = Tuple.Take[T, Tuple.Size[T] - N]
type TakeLast[T <: Tuple, N <: Int] = Tuple.Drop[T, Tuple.Size[T] - N]

type FindIndexT[V, T <: Tuple] <: Int = T match {
  case h *: t => h match {
    case V => 0
    case _ => 1 + FindIndexT[V, t]
  }
}

sealed trait FindIndex[V, T <: Tuple]:
  val value: Int
object FindIndex:
  given found[V, T <: Tuple]: FindIndex[V, V *: T] with 
    override val value = 0
  given search[V, H, T <: Tuple](using ev: FindIndex[V, T]): FindIndex[V, H *: T] with 
    override val value = ev.value + 1

type RemoveIndex[T <: Tuple, I <: Int] <: Tuple = T match {
  case EmptyTuple => EmptyTuple
  case h *: t => I match {
    case 0 => t
    case S[i] => h *: RemoveIndex[t, i]
  }
}

type AddAtIndex[T <: Tuple, A, I <: Int] <: Tuple = T match {
  case EmptyTuple => I match {
    case 0 => A *: EmptyTuple
    case _ => EmptyTuple
  }
  case h *: t => I match {
    case 0 => A *: h *: t
    case S[i] => h *: AddAtIndex[t, A, i]
  }
}

type Swap[T <: Tuple, I1 <: Int, I2 <: Int] = 
  AddAtIndex[
    RemoveIndex[
      AddAtIndex[RemoveIndex[T, I1], Tuple.Elem[T, I2], I1],  // I1 -> I2 
      I2
    ], Tuple.Elem[T, I1], I2 
  ]  // I2 -> I1


trait ValuesOf[T <: Tuple]:
  def value: T
object ValuesOf:
  given ValuesOf[EmptyTuple] with
    val value = EmptyTuple

  given [h, t <: Tuple](using vh: ValueOf[h], vt: ValuesOf[t]): ValuesOf[h *: t] with
    val value = vh.value *: vt.value


@main def main(): Unit =
  
  // Check if T is a dimension
  type Dim[T] = T match {
    case (n -> h) => DummyImplicit
    case (n -= h) => DummyImplicit
  }

  // Checl if T is a tuple of dimensions
  type DimTuple[T <: Tuple] = T match {
    case (n -> h) *: t => DimTuple[t]
    case (n -= h) *: t => DimTuple[t]
    case EmptyTuple => DummyImplicit
  }
 
  case class Tensor[+S <: Tuple](
    val stensor : torch.Tensor[Float32]
  )

  object Tensor:
    def zeros[S <: Tuple : DimTuple](using valueOf: ValuesOf[ExtractShapeTuple[S]]): Tensor[S] =
      val tShape = valueOf.value.asInstanceOf[Tuple].toList.asInstanceOf[Seq[Int]]
      Tensor(torch.zeros(tShape))

  // Type alias for Tensor1 as Tensor[(5)] does not work
  type Tensor1[T] = Tensor[Tuple1[T]]
  object Tensor1:
    def apply[T](): Tensor1[T] = Tensor1[T]()

  // Check if T is a space dimension
  type IsSpace[T] <: Boolean = T match {
    case (_ -> _) => true
    case _ => false
  }

  // Check if T is a data dimension
  type IsData[T] <: Boolean = T match {
    case (_ -= _) => true
    case _ => false
  }

  object TensorOps:

    case class DotProductEvidence[T <: Tuple, U <: Tuple]()

    object DotProductEvidence:
      given dotProductEvidence[T <: Tuple, U <: Tuple](
        using 
        ExtractShape[Tuple.Last[T]] =:= ExtractShape[Tuple.Head[U]],
        // IsSpace[Tuple.Last[T]] =:= true,
        // IsSpace[Tuple.Head[U]] =:= true,
        ThisOrEq[__, ExtractName[Tuple.Last[T]], ExtractName[Tuple.Head[U]]]
      ): DotProductEvidence[T, U] = DotProductEvidence[T, U]()

    type DropNames[T] = T match {
      case _ -> a => __ -> a
      case _ -= a => __ -= a
    }

    type DropNamesTuple[T <: Tuple] = Tuple.Map[T, DropNames]

    // Map F on Shape structure, keeping name structure
    type MapShape[Tup <: Tuple, F[_ <: Tuple.Union[DropNamesTuple[Tup]]]] <: Tuple = Tup match {
      case EmptyTuple => EmptyTuple
      case (n -> h) *: t => (n -> F[h]) *: MapShape[t, F]
      case (n -= h) *: t => (n -= F[h]) *: MapShape[t, F]
      // case h *: t => F[h] *: MapShape[t, F]
    }

    // Zip two shape structures, keeping name structure
    type ZipShape[T1 <: Tuple, T2 <: Tuple] <: Tuple = (T1, T2) match {
      case ((n -> h1) *: t1, (_ -> h2) *: t2) => (n -> (h1, h2)) *: ZipShape[t1, t2]
      case ((n -= h1) *: t1, (_ -= h2) *: t2) => (n -= (h1, h2)) *: ZipShape[t1, t2]
      case (EmptyTuple, EmptyTuple) => EmptyTuple
    }

    case class Conv2dEvidence[Image <: Tuple, Kernel <: Tuple]()

    object Conv2dEvidence:
      given conv2dEvidence[Image <: Tuple, Kernel <: Tuple](
        using
        // channels must match 
        ExtractShape[Tuple.Head[Image]] =:= ExtractShape[Tuple.Head[Kernel]],
        // ThisOrEq[__, ExtractName[Tuple.Last[T]], ExtractName[Tuple.Head[U]]]  // TODO
      ): Conv2dEvidence[Image, Kernel] = Conv2dEvidence[Image, Kernel]()

    trait ConvOps[T <: Tuple, K <: Int, KDim <: ("n_kernels" -= K), U <: Tuple]:
      type Out <: Tuple

    object ConvOps:
      type ConvCalcDim[D] = D match {
        case (s1, s2) => s1 - s2 + 1
      }
      type Aux[T <: Tuple, K <: Int, KDim <: ("n_kernels" -= K), U <: Tuple, O <: Tuple] = ConvOps[T, K, KDim, U] { type Out = O }
      given empty: Aux[EmptyTuple, -1, Nothing, EmptyTuple, EmptyTuple] = new ConvOps[EmptyTuple, -1, Nothing, EmptyTuple] {  type Out = EmptyTuple }

      given convOp[T <: Tuple, K <: Int, KDim <: ("n_kernels" -= K), Kernel <: Tuple](
        using ev: Conv2dEvidence[TakeLast[T, Tuple.Size[Kernel]], Kernel]
      ): Aux[T, K, KDim, Kernel, Tuple.Concat[DropLast[T, Tuple.Size[Kernel]], ("n_channels" -> K) *: Tuple.Tail[MapShape[ZipShape[TakeLast[T, Tuple.Size[Kernel]], Kernel], ConvCalcDim]]]] =
        new ConvOps[T, K, KDim, Kernel] {
          type Out = Tuple.Concat[DropLast[T, Tuple.Size[Kernel]], ("n_channels" -> K) *: Tuple.Tail[MapShape[ZipShape[TakeLast[T, Tuple.Size[Kernel]], Kernel], ConvCalcDim]]]
        }


    @implicitNotFound(msg="DotProductOp not found. Probably shape or names mismatch, shapes are \n ${T} and \n ${U}")
    trait DotProductOp[T <: Tuple, U <: Tuple]:
      type Out <: Tuple

    object DotProductOp:
      type Aux[T <: Tuple, U <: Tuple, O <: Tuple] = DotProductOp[T, U] { type Out = O }
      given empty: Aux[EmptyTuple, EmptyTuple, EmptyTuple] = new DotProductOp[EmptyTuple, EmptyTuple] {  type Out = EmptyTuple }
      given dotProductOp[T <: Tuple, U <: Tuple](
        using ev: DotProductEvidence[T, U]
      ): Aux[T, U, Tuple.Concat[Tuple.Init[T], Tuple.Tail[U]]] =
        new DotProductOp[T, U] {
          type Out = Tuple.Concat[Tuple.Init[T], Tuple.Tail[U]]
        }

    trait AddVectorOp[T <: Tuple, U <: Tuple]:
      type Out <: Tuple

    object AddVectorOp:
      type Aux[T <: Tuple, U <: Tuple, O <: Tuple] = AddVectorOp[T, U] { type Out = O }
      given empty: Aux[EmptyTuple, EmptyTuple, EmptyTuple] = new AddVectorOp[EmptyTuple, EmptyTuple] {  type Out = EmptyTuple }
      given addVectorOp[T <: Tuple, U <: Tuple]: Aux[T, U, T] = new AddVectorOp[T, U] { type Out = T }

    extension [A: Dim, B: Dim](t1: Tensor[(A, B)])
      def t: Tensor[(B, A)] =
        Tensor[(B, A)](t1.stensor.t)

    extension [T <: Tuple](t1: Tensor[T])
      def dot[U <: Tuple](t2: Tensor[U])(using 
        ev: DotProductOp[T, U],
      ): Tensor[ev.Out] = 
        Tensor(torch.matmul(t1.stensor, t2.stensor))

      def *[U <: Tuple](t2: Tensor[U])(using 
        ev: DotProductOp[T, U],
      ): Tensor[ev.Out] = t1.dot(t2)

      def addAlongDim[U <: Tuple](t2: Tensor[U])(using ev: AddVectorOp[T, U]): Tensor[ev.Out] = 
        Tensor(t1.stensor + t2.stensor)

      def +[U <: Tuple](t2: Tensor[U])(using ev: AddVectorOp[T, U]): Tensor[ev.Out] = 
        t1.addAlongDim(t2)

      def conv2d[K <: Int&Singleton, KDim <: ("n_kernels" -= K), U <: Tuple](t2: Tensor[KDim *: U])(using 
        ev: ConvOps[T, K, KDim, U],
      ): Tensor[ev.Out] = 
        println((t1.stensor.shape, t2.stensor.shape))
        Tensor(torch.nn.functional.conv2d(t1.stensor, t2.stensor))
      
      def swapByIndex[Ax1 <: Int, Ax2 <: Int](a1: Ax1, a2: Ax2)(
        using 
        a1.type <= Tuple.Size[T] =:= true,
        a2.type <= Tuple.Size[T] =:= true,
      ): Tensor[Swap[T, a1.type, a2.type]] = Tensor[Swap[T, a1.type, a2.type]](
        torch.swapaxes(t1.stensor, a1, a2)
      )

      def swapByIndexUnsafe[Ax1 <: Int, Ax2 <: Int](a1: Ax1, a2: Ax2)
      : Tensor[Swap[T, a1.type, a2.type]] = Tensor[Swap[T, a1.type, a2.type]](
        torch.swapaxes(t1.stensor, a1, a2)
      )

      def swapByName[Ax1 <: String, Ax2 <: String](a1: Ax1, a2: Ax2)(
        using 
        ev1: FindIndex[a1.type, ExtractNameTuple[T]],
        ev2: FindIndex[a2.type, ExtractNameTuple[T]],
      ): Tensor[Swap[T, FindIndexT[a1.type, ExtractNameTuple[T]], FindIndexT[a2.type, ExtractNameTuple[T]]]] =
        t1.swapByIndexUnsafe(ev1.value, ev2.value).asInstanceOf[
          Tensor[Swap[T, FindIndexT[a1.type, ExtractNameTuple[T]], FindIndexT[a2.type, ExtractNameTuple[T]]]]
        ]

      def dropNames: Tensor[DropNamesTuple[T]] = t1.asInstanceOf[Tensor[DropNamesTuple[T]]]

    def softmax[T <: Tuple](t: Tensor[T]): Tensor[T] = 
      Tensor[T](torch.softmax(t.stensor, 1, dtype = torch.float32))
  
    def relu[T <: Tuple](t: Tensor[T]): Tensor[T] = 
      val relu = torch.nn.ReLU(false)
      Tensor[T](relu(t.stensor))

    
  import TensorOps.*

  /** A named space dimension */
  infix case class ->[+Name <: String, +Shape](name: Name, shape: Shape)

  /** A named data dimension */
  infix case class -=[+Name <: String, +Shape](name: Name, shape: Shape)

  // Extract shape structure from tensor structure
  type ExtractShapeTuple[D <: Tuple] = Tuple.Map[D, ExtractShape]
  type ExtractShape[D] = D match {
    case _ -= s => s
    case _ -> s => s
  }

  // Extract name structure from tensor structure
  type ExtractNameTuple[D <: Tuple] = Tuple.Map[D, ExtractName]
  type ExtractName[D] = D match {
    case d -= s => d
    case d -> s => d
  }

  // Example of a higher library abstraction on top of tensor-safety
  case class Dense[
    From <: __ -> ?, To <: __ -> ?
  ](
      w1: Tensor[(From, To)], 
      b1: Tensor[Tuple1[To]],
  ):
    type _From = From
    type _To = To
    def apply[BS <: Int & Singleton](
      x: Tensor[(__ -= BS, From)],
      activation: [T <: Tuple] => Tensor[T] => Tensor[T] = [T <: Tuple] => (x: Tensor[T]) => relu[T](x)
    ): Tensor[(__ -= BS, To)] = 
      val h = x*w1 + b1
      return activation(h)

  object Dense:
    def random[A <: Int, B <: Int](
      using ValueOf[A], ValueOf[B]
    ): Dense[__ -> A, __ -> B] = 
      Dense(
        w1=Tensor.zeros[(__ -> A, __ -> B)],
        b1=Tensor.zeros[Tuple1[__ -> B]]
      )

  case class ModelWithDense(
    val l1 = Dense.random[28*28, 128]
    val l2 = Dense.random[128, 10]
  ):
    def apply[
      BS <: Int & Singleton
    ](
      x: Tensor[(__ -= BS, l1._From)]
    ): Tensor[(__ -= BS, l2._To)] = 
      val x2 = l1(x)
      l2(x2)

  case class Model(
      w1: Tensor[(__ -> 28*28, __ -> 128)], 
      b1: Tensor1[__ -> 128], 
      w2: Tensor[(__ -> 128, __ -> 10)], 
      b2 : Tensor1[__ -> 10]
  ): 
      def apply[BatchSize <: Int](
        x: Tensor[(__ -= BatchSize, __ -> 28*28)]
      ): Tensor[(__ -= BatchSize, __ -> 10)] = 
        val h = relu(x*w1+b1)
        softmax(h*w2+b2)

  case class CNNModel[
    L1 <: Int&Singleton, L2 <: Int&Singleton
  ](
      kernel1: Tensor[("n_kernels" -= L1, "n_channels" -> 1, __ -= 3, __ -= 3)], 
      kernel2: Tensor[("n_kernels" -= L2, "n_channels" -> L1, __ -= 3, __ -= 3)], 
  ):
    
    def apply[BS <: Int&Singleton, S1 <: String, S2 <: String, S3 <: String, S4 <: String](x: Tensor[(S1 -= BS, S4 -> 1, S2 -= 28, S3 -= 28)]) = 
      val x2 = x.conv2d(kernel1)
      val x3 = x2.conv2d(kernel2)
      x.conv2d(kernel1)
       .conv2d(kernel2)

  // Examples
  {
    println("Model")
    val w1 = Tensor.zeros[(__ -> 28*28, __ -> 128)]
    val b1 = Tensor.zeros[Tuple1[__ -> 128]]
    val w2 = Tensor.zeros[(__ -> 128, __ -> 10)]
    val b2 = Tensor.zeros[Tuple1[__ -> 10]]
    val m = Model(w1, b1, w2, b2)

    val x = Tensor.zeros[(__ -= 512, __ -> 28*28)]
    val y = m(x)
  }
  {
    println("ModelWithDense")
    val m = ModelWithDense()
    val x = Tensor.zeros[(__ -= 512, __ -> 28*28)]
    val y = m(x)
    println(y.stensor.shape)
  }
  {
    println("swapaxes")
    val x1 = Tensor.zeros[(__ -= 1, __ -= 2, __ -= 3, __ -= 4)]
    val x2 = x1.swapByIndex(1, 2)
  }
  {
    println("swap by name")
    val x1 = Tensor.zeros[("a" -= 1, __ -= 2, "b" -= 3, __ -= 4)]
    val x2 = x1.swapByName("a", "b")
    val x3 = x2.dropNames
  }
  {
    println("CNNModel")
    val m = CNNModel(
      kernel1=Tensor.zeros[("n_kernels" -= 32, "n_channels" -> 1, __ -= 3, __ -= 3)],
      kernel2=Tensor.zeros[("n_kernels" -= 32, "n_channels" -> 32, __ -= 3, __ -= 3)],
    )
    val x1 = Tensor.zeros[(__ -= 512, __ -> 1, __ -= 28, __ -= 28)]
    val y1 = m(x1)

    val z1 = y1.conv2d(m.kernel2)
    val z2 = z1.conv2d(m.kernel2)
    val z3 = z2.conv2d(m.kernel2)
    val z4 = z3.conv2d(m.kernel2)

    val x2 = Tensor.zeros[(__ -= 512, "n_channels" -> 1, "width" -= 28, "height" -= 28)]
    val x3 = Tensor.zeros[(__ -= 512,  __ -> 1, "width" -= 28, "height" -= 28)]
    val y2 = m(x2)
    val y3 = m(x3)

    val X = Tensor.zeros[("_" -= 10_000, "_" -> 10)]
    val Cov = X * X.t
    val CrossCov = X.t * X
  }