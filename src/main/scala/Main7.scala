package main7

import scala.compiletime.ops.int._
import java.util.jar.Attributes.Name
import scala.compiletime.{erasedValue, constValue, constValueOpt}
import scala.util.Random
import scala.annotation.targetName
import torch.Float32
import scala.annotation.implicitNotFound
import spire.random.Size
import torch.nn.functional.conv2d

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


inline def constValueTuple[T <: Tuple]: T =
  (inline erasedValue[T] match
    case _: EmptyTuple => EmptyTuple
    case _: (t *: ts) =>
      // check for recursive tuples
      (constValueOpt[t] match
        case Some(v) => v *: constValueTuple[ts]
        case None =>
          inline erasedValue[t] match
            case _: (t2 *: ts2) => constValueTuple[t2 *: ts2].asInstanceOf[t] *: constValueTuple[ts]
            case _ => -1 *: constValueTuple[ts]
      ).asInstanceOf[t *: ts]
  ).asInstanceOf[T]

@main def hello6(): Unit =
  
  case class Tensor[S <: Tuple](
    val stensor : torch.Tensor[Float32]
  )

  object Tensor:
    def zeros[S <: Tuple](using valueOf: ValuesOf[ExtractShapeTuple[S]]): Tensor[S] =
      val tShape = valueOf.value.asInstanceOf[Tuple].toList.asInstanceOf[Seq[Int]]
      Tensor(torch.zeros(tShape))

    def zeros(s: Tuple): Tensor[EmptyTuple] = 
      val tShape = s.toList.asInstanceOf[Seq[Int]]
      Tensor(torch.zeros(tShape))

  type Tensor1[T] = Tensor[Tuple1[T]]

  object TensorOps:

    case class DotProductEvidence[T <: Tuple, U <: Tuple]()

    object DotProductEvidence:
      given dotProductEvidence[T <: Tuple, U <: Tuple](
        using 
        ExtractShape[Tuple.Last[T]] =:= ExtractShape[Tuple.Head[U]],
        ThisOrEq["_unnamed_", ExtractName[Tuple.Last[T]], ExtractName[Tuple.Head[U]]]
      ): DotProductEvidence[T, U] = DotProductEvidence[T, U]()

    type DropNames[T] = T match {
      case _ -> a => -?>[a]
      case _ -= a => -?=[a]
      case _ => T
    }

    type DropNamesTuple[T <: Tuple] = Tuple.Map[T, DropNames]

    type MapShape[Tup <: Tuple, F[_ <: Tuple.Union[DropNamesTuple[Tup]]]] <: Tuple = Tup match {
      case EmptyTuple => EmptyTuple
      case (n -> h) *: t => (n -> F[h]) *: MapShape[t, F]
      case (n -= h) *: t => (n -= F[h]) *: MapShape[t, F]
      case h *: t => F[h] *: MapShape[t, F]
    }

    type ZipShape[T1 <: Tuple, T2 <: Tuple] <: Tuple = (T1, T2) match {
      case ((n -> h1) *: t1, h2 *: t2) => (n -> (h1, h2)) *: ZipShape[t1, t2]
      case ((n -= h1) *: t1, h2 *: t2) => (n -= (h1, h2)) *: ZipShape[t1, t2]
      case (h1 *: t1, h2 *: t2) => (h1, h2) *: ZipShape[t1, t2]
      case _ => EmptyTuple
    }

    
    trait ConvOps[T <: Tuple, U <: Tuple]:
      type Out <: Tuple

    object ConvOps:
      type ConvCalcDim[D] = D match {
        case (s1, s2) => s1 - s2 + 1
      }
      type Aux[T <: Tuple, U <: Tuple, O <: Tuple] = ConvOps[T, U] { type Out = O }
      given empty: Aux[EmptyTuple, EmptyTuple, EmptyTuple] = new ConvOps[EmptyTuple, EmptyTuple] {  type Out = EmptyTuple }

      given convOp[T <: Tuple, U <: Tuple](
        // using ev: DotProductEvidence[T, U]
      ): Aux[T, U, Tuple.Concat[DropLast[T, Tuple.Size[U]], MapShape[ZipShape[TakeLast[T, Tuple.Size[U]], U], ConvCalcDim]]] =
        new ConvOps[T, U] {
          type Out = Tuple.Concat[DropLast[T, Tuple.Size[U]], MapShape[ZipShape[TakeLast[T, Tuple.Size[U]], U], ConvCalcDim]]
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

    extension [A, B](t1: Tensor[(A, B)])
      def t: Tensor[(B, A)] =
        Tensor[(B, A)](t1.stensor.t)

    extension [T <: Tuple](t1: Tensor[T])
      def dot[U <: Tuple](t2: Tensor[U])(using 
        ev: DotProductOp[T, U],
      ): Tensor[ev.Out] = 
        Tensor(torch.matmul(t1.stensor, t2.stensor))
      def addAlongDim[U <: Tuple](t2: Tensor[U])(using ev: AddVectorOp[T, U]): Tensor[ev.Out] = 
        Tensor(t1.stensor + t2.stensor)

      def conv2d[U <: Tuple](t2: Tensor[U])(using 
        ev: ConvOps[T, U],
      ): Tensor[ev.Out] = 
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

  object Tensor1:
    def apply[T](): Tensor1[T] = Tensor1[T]()

  type Shape[D] = D match {
    case Int => D
    case (D, _) => D
  }
  
  /** A named space dimension */
  infix case class ->[Name <: String, Shape](name: Name, shape: Shape)
  case class -?>[Shape](shape: Shape)

  /** A named data dimension */
  infix case class -=[Name <: String, Shape](name: Name, shape: Shape)
  case class -?=[Shape](shape: Shape)

  type ExtractShapeTuple[D <: Tuple] = Tuple.Map[D, ExtractShape]

  type ExtractShape[D] = D match {
    case d -= s => s
    case d -> s => s
    case -?=[s] => s
    case -?>[s] => s
    case Int => D
  }

  type ExtractNameTuple[D <: Tuple] = Tuple.Map[D, ExtractName]

  type ExtractName[D] = D match {
    case d -= s => d
    case d -> s => d
    case _ => "_unnamed_"
  }

  case class PythoneskModel(
      w1: Tensor[EmptyTuple], 
      b1: Tensor[EmptyTuple], 
      w2: Tensor[EmptyTuple], 
      b2 : Tensor[EmptyTuple]
  ): 
    
    @targetName("apply named")
    def apply(
      x: Tensor[EmptyTuple]
    ): Tensor[EmptyTuple] = 
      val h1 = relu(x.dot(w1)).addAlongDim(b1)
      val out = h1.dot(w2).addAlongDim(b2)
      softmax(out)

  case class Model2(
      w1: Tensor[(28*28, 128)], 
      b1: Tensor1[128], 
      w2: Tensor[(128, 10)], 
      b2 : Tensor1[10]
  ): 
      
      @targetName("apply named")
      def apply[NS <: Int, N <: String](
        x: Tensor[(NS, 28*28)]
      ): Tensor[(NS, 10)] = 
        val h1 = relu(x.dot(w1)).addAlongDim(b1)
        val out = h1.dot(w2).addAlongDim(b2)
        softmax(out) 


  case class Model[
    NP <: Int, H <: Int, NC <: Int
  ](
      w1: Tensor[("pixels" -> NP, H)], 
      b1: Tensor1[H], 
      w2: Tensor[(H, NC)], 
      b2 : Tensor1[NC]
  ): 
    
    @targetName("apply named")
    def apply[NS <: Int, Batch <: String](
      x: Tensor[(Batch -= NS, "pixels" -> NP)]
    ): Tensor[(Batch -= NS, "logits" -> NC)] =
      val h1 = relu(x.dot(w1)).addAlongDim(b1)
      val out = h1.dot(w2).addAlongDim(b2)
      val res = softmax(out)
      Tensor[(Batch -= NS, "logits" -> NC)](res.stensor)   

    @targetName("apply unnamed")
    def apply[NS <: Int](
      x: Tensor[(NS, NP)]
    ): Tensor[(NS, NC)] = 
      val h1 = relu(x.dot(w1)).addAlongDim(b1)
      val out = h1.dot(w2).addAlongDim(b2)
      softmax(out) 

  case class CNNModelGeneral[
    W <: Int, H <: Int
  ](
      kernel1: Tensor[(W, H, 1)], 
      kernel2: Tensor[(W, H, 1)], 
  ): 
    
    def apply[T <: Tuple](
      x: Tensor[T]
    )(using TakeLast[ExtractShapeTuple[T], 3] =:= (28, 28, 1)) = 
      val x2 = x.conv2d(kernel1)
      x2.conv2d(kernel2)

  case class CNNModel(
      kernel1: Tensor[(3, 3, 1)], 
      kernel2: Tensor[(3, 3, 1)], 
  ):
    
    def apply[BS <: Int](x: Tensor[(BS, 28, 28, 1)]) = 
      val x2 = x.conv2d(kernel1)
      x.conv2d(kernel1)
       .conv2d(kernel2)

  val w1 = Tensor.zeros[("pixels" -> 28*28, 128)]
  val b1 = Tensor.zeros[Tuple1[128]]
  val w2 = Tensor.zeros[(128, 10)]
  val b2 = Tensor.zeros[Tuple1[10]]
  val m = Model(w1, b1, w2, b2)

  {
    println("extra named dimensions Model")
    // With extra named dimensions
    val x = Tensor.zeros[("samples" -= 128, "pixels" -> 28*28)]
    val y = m(x)                     // val y: Tensor[("samples" -= 128, "logits" -> 10)]
    println(y)
  }
  {
    println("no named dimensions Model")
    // Without extra named dimensions
    val x = Tensor.zeros[(128, 28*28)]
    val y = m(x)                     // val y: Tensor[(128, 10)]
    println(y)
  }
  {
    println("Pythonesk Model")
    val w1 = Tensor.zeros((28*28, 128))
    val b1 = Tensor.zeros(Tuple1(128))
    val w2 = Tensor.zeros((128, 10))
    val b2 = Tensor.zeros(Tuple1(10))
    val m = PythoneskModel(w1, b1, w2, b2)
    val x = Tensor.zeros[(128, 28*28)].asInstanceOf[Tensor[EmptyTuple]]
    val y = m(x)                     // val y: Tensor[(128, 10)]
    println(y)
  }
  {
    println("swapaxes")
    val x1 = Tensor.zeros[(1, 2, 3, 4)]
    val x2 = x1.swapByIndex(1, 2)
  }
  {
    println("swap by name")
    val x1 = Tensor.zeros[("a" -> 1, 2, "b" -> 3, 4)]
    val x2 = x1.swapByName("a", "b")
    val x3 = x2.dropNames
  }
  {
    println("CNNModel")
    val m = CNNModelGeneral(
      kernel1=Tensor.zeros[(3,3,1)],
      kernel2=Tensor.zeros[(3,3,1)],
    )
    val x1 = Tensor.zeros[(512, 28, 28, 1)]
    val x2 = Tensor.zeros[(512, "width" -= 28, "height" -= 28, "channel" -> 1)]
    val x3 = Tensor.zeros[(512, "width" -= 28, "height" -= 28, 1)]

    val y1 = m(x1)
    val y2 = m(x2)
    val y3 = m(x3)

    val X = Tensor.zeros[(-?=[10_000], -?>[10])]
    val Cov = X.dot(X.t)
    val CrossCov = X.t.dot(X)
  }