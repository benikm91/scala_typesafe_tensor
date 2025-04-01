package main6

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

  type Tensor1[T] = Tensor[Tuple1[T]]

  object TensorOps:

    case class DotProductEvidence[T <: Tuple, U <: Tuple]()

    object DotProductEvidence:
      given dotProductEvidence[T <: Tuple, U <: Tuple](
        using 
        ExtractShape[Tuple.Last[T]] =:= ExtractShape[Tuple.Head[U]],
        ThisOrEq["_unnamed_", ExtractName[Tuple.Last[T]], ExtractName[Tuple.Head[U]]]
      ): DotProductEvidence[T, U] = DotProductEvidence[T, U]()

    type CalcDim[D] = D match {
      case (d1 -> s1, s2) => d1 -> (s1 - s2 + 1)  // this must be "automatic" for all named cases
      case (d1 -= s1, s2) => d1 -= (s1 - s2 + 1)  // this must be "automatic" for all named cases
      case (s1, s2) => s1 - s2 + 1
    }
    
    trait ConvOps[T <: Tuple, U <: Tuple]:
      type Out <: Tuple

    object ConvOps:
      type Aux[T <: Tuple, U <: Tuple, O <: Tuple] = ConvOps[T, U] { type Out = O }
      given empty: Aux[EmptyTuple, EmptyTuple, EmptyTuple] = new ConvOps[EmptyTuple, EmptyTuple] {  type Out = EmptyTuple }

      given convOp[T <: Tuple, U <: Tuple](
        // using ev: DotProductEvidence[T, U]
      ): Aux[T, U, Tuple.Concat[DropLast[T, Tuple.Size[U]], Tuple.Map[Tuple.Zip[TakeLast[T, Tuple.Size[U]], U], CalcDim]]] =
        new ConvOps[T, U] {
          type Out = Tuple.Concat[DropLast[T, Tuple.Size[U]], Tuple.Map[Tuple.Zip[TakeLast[T, Tuple.Size[U]], U], CalcDim]]
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

    extension [T <: Tuple](t1: Tensor[T])
      def dot[U <: Tuple](t2: Tensor[U])(using 
        ev: DotProductOp[T, U],
      ): Tensor[ev.Out] = 
        Tensor(torch.matmul(t1.stensor, t2.stensor))
      def addAlongDim[U <: Tuple](t2: Tensor[U])(using ev: AddVectorOp[T, U]): Tensor[ev.Out] = 
        Tensor(t1.stensor + t2.stensor)

      def conv[U <: Tuple](t2: Tensor[U])(using 
        ev: ConvOps[T, U],
      ): Tensor[ev.Out] = 
        Tensor(conv2d(t1.stensor, t2.stensor))

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
  infix case class ->[Name <: String, Shape <: Int](name: Name, shape: Shape)

  /** A named data dimension */
  infix case class -=[Name <: String, Shape <: Int](name: Name, shape: Shape)

  type ExtractShapeTuple[D <: Tuple] = Tuple.Map[D, ExtractShape]

  type ExtractShape[D] = D match {
    case d -= s => s
    case d -> s => s
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

  case class CNNModel[
    W <: Int, H <: Int
  ](
      kernel1: Tensor[(W, H, 1)], 
      kernel2: Tensor[(W, H, 1)], 
  ): 
    
    def apply[T <: Tuple](
      x: Tensor[T]
    )(using ExtractShapeTuple[T] =:= (28, 28, 1)) = x
        .conv(kernel1)
        .conv(kernel2)


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
    val w1 = Tensor.zeros[(28*28, 128)].asInstanceOf[Tensor[EmptyTuple]]
    val b1 = Tensor.zeros[Tuple1[128]].asInstanceOf[Tensor[EmptyTuple]]
    val w2 = Tensor.zeros[(128, 10)].asInstanceOf[Tensor[EmptyTuple]]
    val b2 = Tensor.zeros[Tuple1[10]].asInstanceOf[Tensor[EmptyTuple]]
    val m = PythoneskModel(w1, b1, w2, b2)
    val x = Tensor.zeros[(128, 28*28)].asInstanceOf[Tensor[EmptyTuple]]
    val y = m(x)                     // val y: Tensor[(128, 10)]
    println(y)
  }
  {
    println("CNNModel")
    val m = CNNModel(
      kernel1=Tensor.zeros[(3,3,1)],
      kernel2=Tensor.zeros[(3,3,1)],
    )
    val x1 = Tensor.zeros[(28, 28, 1)]
    val x2 = Tensor.zeros[("width" -= 28, "height" -= 28, "channel" -> 1)]
    val x3 = Tensor.zeros[("width" -= 28, "height" -= 28, 1)]
    val y1 = m(x1)
    val y2 = m(x2)
    val y3 = m(x3)
    println(y1)
  }