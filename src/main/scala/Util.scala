package util

import scala.compiletime.{constValue, constValueOpt}
import scala.compiletime.ops.int
import scala.compiletime.ops.int._
import scala.compiletime.ops.boolean._
import scala.quoted.*
import scala.compiletime.erasedValue
import scala.compiletime.summonInline
import scala.compiletime.ops.int.S
import scala.Tuple.{Concat, Head, Init, Last, Tail, Elem}


type Dims = Tuple

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
      AddAtIndex[RemoveIndex[T, I1], Elem[T, I2], I1],  // I1 -> I2 
      I2
    ], Elem[T, I1], I2 
  ]  // I2 -> I1

type Length[T <: Tuple] <: Int = T match {
  case EmptyTuple => 0
  case h *: t => 1 + Length[t]
}

sealed trait LTE[A <: Int, B <: Int]
object LTE {
  class LTEImpl[A <: Int, B <: Int] extends LTE[A, B]

  given LTEImpl[0, 1]()
  
  inline given [A <: Int, B <: Int](using ev: LTE[A, B]): LTE[A, S[B]] = 
    LTEImpl[A, S[B]]

  inline given [A <: Int, B <: Int](using ev: LTE[A, B]): LTE[S[A], S[B]] = 
    LTEImpl[S[A], S[B]]

  def apply[A <: Int, B <: Int](using ev: LTE[A, B]): LTE[A, B] = ev

}

type IntTuple[T <: Tuple] = T match
  case Int *: t => IntTuple[t]
  case EmptyTuple => DummyImplicit

inline def tupleShape[T <: Tuple]: T = constValueTuple[T]

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