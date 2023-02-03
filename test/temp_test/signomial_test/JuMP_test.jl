using OrderedCollections

relaxed_tols = (default_tol_relax = 100,)
insts = OrderedDict()
insts["minimal"] = [
    ((2, 2),),
    ((2, 2), :SOCExpPSD),
    ]
insts["fast"] = [
    ((:motzkin2,),),
    ((:motzkin2,), :SOCExpPSD),
    ((:motzkin3,),),
    ((:CS16ex8_13,),),
    ((:CS16ex8_14,),),
    ((:CS16ex18,),),
    ((:CS16ex12,),),
    ((:CS16ex13,),),
    ((:MCW19ex1_mod,),),
    ((:MCW19ex8,),),
    ((:MCW19ex8,), :SOCExpPSD),
    ((3, 2),),
    ((3, 2), :SOCExpPSD),
    ((6, 6),),
    ((20, 3),),
    ((20, 3), :SOCExpPSD),
    ]
insts["various"] = [
    ((:motzkin2,),),
    ((:motzkin3,),),
    ((:CS16ex8_13,),),
    ((:CS16ex8_14,),),
    ((:CS16ex18,),),
    ((:CS16ex12,),),
    ((:CS16ex13,),),
    ((:MCW19ex1_mod,),),
    ((:MCW19ex8,), nothing, relaxed_tols),
    ((6, 3),),
    ((6, 3), :SOCExpPSD),
    ((20, 3), nothing, relaxed_tols),
    ((20, 3), :SOCExpPSD, relaxed_tols),
    ]
return (SignomialMinJuMP, insts)
