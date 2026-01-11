(define (domain npuzzle)
  (:requirements :strips :typing)
  (:types tile pos)

  (:predicates
    (at ?t - tile ?p - pos)
    (blank ?p - pos)
    (adj ?p1 - pos ?p2 - pos)
  )

  (:action move
    :parameters (?t - tile ?from - pos ?to - pos)
    :precondition (and
      (at ?t ?from)
      (blank ?to)
      (adj ?from ?to)
    )
    :effect (and
      (not (at ?t ?from))
      (at ?t ?to)
      (not (blank ?to))
      (blank ?from)
    )
  )
)
