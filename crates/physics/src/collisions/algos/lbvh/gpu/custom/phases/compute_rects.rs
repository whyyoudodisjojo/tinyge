use codegen::asts::lowered::{
    Accessor, BinOp, LoweredAST, Scope, VarRef, VarRefType,
};
use codegen::asts::lowered::BindedBuffer;
use codegen::dt::{BasicTy, DType, IntegerTy, VecTy};
use codegen_macros::{IntoWgslStruct, shader};

#[derive(IntoWgslStruct)]
pub struct Vertex {
    pos: [f32; 3],
    _pad: f32,
}

#[derive(IntoWgslStruct)]
pub struct ModelInfo {
    offset: u32,
    stride: u32,
}

#[derive(IntoWgslStruct)]
pub struct RectangleBounds {
    min: [f32; 3],
    max: [f32; 3],
}

#[shader(compute(workgroup_sz = 256))]
fn compute_rects(
    #[binding(storage(read_only = true))] model_verts: BindedBuffer<Vertex, 0>,
    #[binding(storage(read_only = true))] model_infos: BindedBuffer<ModelInfo, 1>,
    #[binding(storage(read_only = false))] output_rect: BindedBuffer<RectangleBounds, 2>,
    #[shared] sdata_min: [f32; 3],
    #[shared] sdata_max: [f32; 3],
) -> Scope {
    let mut scope = Scope::new();

    let lid = scope.add_local(
        "lid".to_string(), false,
        LoweredAST::Load(VarRefType::EntryPointGlobal(VarRef {
            id: 1,
            by: vec![Accessor::Field("x".to_string())],
        })),
    );
    let model_idx = scope.add_local(
        "model_idx".to_string(), false,
        LoweredAST::Load(VarRefType::EntryPointGlobal(VarRef {
            id: 0,
            by: vec![Accessor::Field("x".to_string())],
        })),
    );
    let info = scope.add_local(
        "info".to_string(), false,
        LoweredAST::Load(
            model_infos.var_ref().index(
                LoweredAST::Load(VarRefType::Local(VarRef { id: model_idx, by: vec![] }))
            )
        ),
    );
    let model_offset = scope.add_local(
        "model_offset".to_string(), false,
        LoweredAST::Load(
            VarRefType::Local(VarRef { id: info, by: vec![] })
                .field("offset")
        ),
    );
    let model_vertex_count = scope.add_local(
        "model_vertex_count".to_string(), false,
        LoweredAST::Load(
            VarRefType::Local(VarRef { id: info, by: vec![] })
                .field("stride")
        ),
    );
    let local_min = scope.add_local(
        "local_min".to_string(), true,
        LoweredAST::Const {
            dt: DType::Vector(VecTy::Vec3(BasicTy::F32)),
            data: f32::INFINITY.to_le_bytes().into_iter()
                .chain(f32::INFINITY.to_le_bytes())
                .chain(f32::INFINITY.to_le_bytes())
                .collect(),
        },
    );
    let local_max = scope.add_local(
        "local_max".to_string(), true,
        LoweredAST::Const {
            dt: DType::Vector(VecTy::Vec3(BasicTy::F32)),
            data: (-f32::INFINITY).to_le_bytes().into_iter()
                .chain((-f32::INFINITY).to_le_bytes())
                .chain((-f32::INFINITY).to_le_bytes())
                .collect(),
        },
    );
    let i = scope.add_local(
        "i".to_string(), true,
        LoweredAST::Load(VarRefType::Local(VarRef { id: lid, by: vec![] })),
    );
    let offset = scope.add_local(
        "offset".to_string(), true,
        LoweredAST::Const {
            dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
            data: 128u32.to_le_bytes().to_vec(),
        },
    );

    scope.ast = Some(LoweredAST::Group(vec![
        scope.while_loop(
            LoweredAST::BinaryOp {
                lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                    id: model_vertex_count, by: vec![],
                }))),
                rhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                    id: i, by: vec![],
                }))),
                op: BinOp::Gt,
            },
            |b| {
                let v = b.add_local(
                    "v".to_string(), false,
                    LoweredAST::Load(
                        model_verts.var_ref().index(
                            LoweredAST::BinaryOp {
                                lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                    id: model_offset, by: vec![],
                                }))),
                                rhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                    id: i, by: vec![],
                                }))),
                                op: BinOp::Add,
                            },
                        ).field("pos")
                    ),
                );
                b.ast = Some(LoweredAST::Group(vec![
                    LoweredAST::Store {
                        var: VarRefType::Local(VarRef { id: local_min, by: vec![] }),
                        val: Box::new(LoweredAST::FunctionCall {
                            ident: "min".to_string(),
                            args: vec![
                                Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                    id: local_min, by: vec![],
                                }))),
                                Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                    id: v, by: vec![],
                                }))),
                            ],
                        }),
                    },
                    LoweredAST::Store {
                        var: VarRefType::Local(VarRef { id: local_max, by: vec![] }),
                        val: Box::new(LoweredAST::FunctionCall {
                            ident: "max".to_string(),
                            args: vec![
                                Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                    id: local_max, by: vec![],
                                }))),
                                Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                    id: v, by: vec![],
                                }))),
                            ],
                        }),
                    },
                    LoweredAST::Store {
                        var: VarRefType::Local(VarRef { id: i, by: vec![] }),
                        val: Box::new(LoweredAST::BinaryOp {
                            lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                id: i, by: vec![],
                            }))),
                            rhs: Box::new(LoweredAST::Const {
                                dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                                data: 256u32.to_le_bytes().to_vec(),
                            }),
                            op: BinOp::Add,
                        }),
                    },
                ]));
            },
        ),
        LoweredAST::Store {
            var: VarRefType::Shared(VarRef {
                id: 0,
                by: vec![Accessor::Index(Box::new(
                    LoweredAST::Load(VarRefType::Local(VarRef { id: lid, by: vec![] })),
                ))],
            }),
            val: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                id: local_min, by: vec![],
            }))),
        },
        LoweredAST::Store {
            var: VarRefType::Shared(VarRef {
                id: 1,
                by: vec![Accessor::Index(Box::new(
                    LoweredAST::Load(VarRefType::Local(VarRef { id: lid, by: vec![] })),
                ))],
            }),
            val: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                id: local_max, by: vec![],
            }))),
        },
        LoweredAST::FunctionCall {
            ident: "workgroupBarrier".to_string(), args: vec![],
        },
        scope.while_loop(
            LoweredAST::BinaryOp {
                lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                    id: offset, by: vec![],
                }))),
                rhs: Box::new(LoweredAST::Const {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: 0u32.to_le_bytes().to_vec(),
                }),
                op: BinOp::Gt,
            },
            |b| {
                let if_ast = b.cond(
                    LoweredAST::BinaryOp {
                        lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                            id: offset, by: vec![],
                        }))),
                        rhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                            id: lid, by: vec![],
                        }))),
                        op: BinOp::Gt,
                    },
                    |ib| {
                        ib.ast = Some(LoweredAST::Group(vec![
                            LoweredAST::Store {
                                var: VarRefType::Shared(VarRef {
                                    id: 0,
                                    by: vec![Accessor::Index(Box::new(
                                        LoweredAST::Load(VarRefType::Local(VarRef {
                                            id: lid, by: vec![],
                                        })),
                                    ))],
                                }),
                                val: Box::new(LoweredAST::FunctionCall {
                                    ident: "min".to_string(),
                                    args: vec![
                                        Box::new(LoweredAST::Load(VarRefType::Shared(VarRef {
                                            id: 0,
                                            by: vec![Accessor::Index(Box::new(
                                                LoweredAST::Load(VarRefType::Local(VarRef {
                                                    id: lid, by: vec![],
                                                })),
                                            ))],
                                        }))),
                                        Box::new(LoweredAST::Load(VarRefType::Shared(VarRef {
                                            id: 0,
                                            by: vec![Accessor::Index(Box::new(
                                                LoweredAST::BinaryOp {
                                                    lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                                        id: lid, by: vec![],
                                                    }))),
                                                    rhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                                        id: offset, by: vec![],
                                                    }))),
                                                    op: BinOp::Add,
                                                },
                                            ))],
                                        }))),
                                    ],
                                }),
                            },
                            LoweredAST::Store {
                                var: VarRefType::Shared(VarRef {
                                    id: 1,
                                    by: vec![Accessor::Index(Box::new(
                                        LoweredAST::Load(VarRefType::Local(VarRef {
                                            id: lid, by: vec![],
                                        })),
                                    ))],
                                }),
                                val: Box::new(LoweredAST::FunctionCall {
                                    ident: "max".to_string(),
                                    args: vec![
                                        Box::new(LoweredAST::Load(VarRefType::Shared(VarRef {
                                            id: 1,
                                            by: vec![Accessor::Index(Box::new(
                                                LoweredAST::Load(VarRefType::Local(VarRef {
                                                    id: lid, by: vec![],
                                                })),
                                            ))],
                                        }))),
                                        Box::new(LoweredAST::Load(VarRefType::Shared(VarRef {
                                            id: 1,
                                            by: vec![Accessor::Index(Box::new(
                                                LoweredAST::BinaryOp {
                                                    lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                                        id: lid, by: vec![],
                                                    }))),
                                                    rhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                                        id: offset, by: vec![],
                                                    }))),
                                                    op: BinOp::Add,
                                                },
                                            ))],
                                        }))),
                                    ],
                                }),
                            },
                        ]));
                    },
                    None::<fn(&mut Scope)>,
                );
                b.ast = Some(LoweredAST::Group(vec![
                    if_ast,
                    LoweredAST::FunctionCall {
                        ident: "workgroupBarrier".to_string(), args: vec![],
                    },
                    LoweredAST::Store {
                        var: VarRefType::Local(VarRef { id: offset, by: vec![] }),
                        val: Box::new(LoweredAST::BinaryOp {
                            lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                                id: offset, by: vec![],
                            }))),
                            rhs: Box::new(LoweredAST::Const {
                                dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                                data: 1u32.to_le_bytes().to_vec(),
                            }),
                            op: BinOp::Shr,
                        }),
                    },
                ]));
            },
        ),
        scope.cond(
            LoweredAST::BinaryOp {
                lhs: Box::new(LoweredAST::Load(VarRefType::Local(VarRef {
                    id: lid, by: vec![],
                }))),
                rhs: Box::new(LoweredAST::Const {
                    dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                    data: 0u32.to_le_bytes().to_vec(),
                }),
                op: BinOp::Eq,
            },
            |b| {
                b.ast = Some(LoweredAST::Group(vec![
                    LoweredAST::Store {
                        var: output_rect.var_ref().index(
                            LoweredAST::Load(VarRefType::Local(VarRef {
                                id: model_idx, by: vec![],
                            })),
                        ).field("min"),
                        val: Box::new(LoweredAST::Load(VarRefType::Shared(VarRef {
                            id: 0,
                            by: vec![Accessor::Index(Box::new(LoweredAST::Const {
                                dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                                data: 0u32.to_le_bytes().to_vec(),
                            }))],
                        }))),
                    },
                    LoweredAST::Store {
                        var: output_rect.var_ref().index(
                            LoweredAST::Load(VarRefType::Local(VarRef {
                                id: model_idx, by: vec![],
                            })),
                        ).field("max"),
                        val: Box::new(LoweredAST::Load(VarRefType::Shared(VarRef {
                            id: 1,
                            by: vec![Accessor::Index(Box::new(LoweredAST::Const {
                                dt: DType::Basic(BasicTy::Integer(IntegerTy::U32)),
                                data: 0u32.to_le_bytes().to_vec(),
                            }))],
                        }))),
                    },
                ]));
            },
            None::<fn(&mut Scope)>,
        ),
    ]));

    scope
}
