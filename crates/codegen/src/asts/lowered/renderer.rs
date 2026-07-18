use crate::{
    asts::lowered::{
        Accessor, BinOp, CustomBufferBindingType, EntrypointData,
        LoweredAST::{self},
        ShaderIR, Struct, UnaryOp, VarRefType,
        scope::Scope,
    },
    dt::{BasicTy, BasicTyOrStructRef, DType, IntegerTy, MaybeAtomic, VecTy},
};

pub trait Render {
    type Args;
    fn render(&self, args: Self::Args) -> String;
}

pub struct LoweredRenderer<'a> {
    pub ir: &'a ShaderIR,
}

impl<'a> LoweredRenderer<'a> {
    pub fn render_dtype(&self, dt: &DType) -> String {
        match dt {
            DType::Atomic(int_ty) => format!("atomic<{}>", self.render_integer_ty(int_ty)),
            DType::Basic(b) => self.render_basic_ty(b),
            DType::Vector(v) => match v {
                VecTy::Array(inner) => format!("array<{}>", self.render_array_inner(inner)),
                VecTy::Vec2(b) => format!("vec2<{}>", self.render_basic_ty(b)),
                VecTy::Vec3(b) => format!("vec3<{}>", self.render_basic_ty(b)),
                VecTy::Vec4(b) => format!("vec4<{}>", self.render_basic_ty(b)),
            },
            DType::StructRef { ident } => ident.clone(),
            DType::Pad(bytes) => self.render_pad_type(*bytes),
        }
    }

    fn render_pad_type(&self, bytes: usize) -> String {
        match bytes {
            4 => "u32".to_string(),
            8 => "vec2<u32>".to_string(),
            12 => "vec3<u32>".to_string(),
            16 => "vec4<u32>".to_string(),
            n => panic!(
                "Unsupported padding size: {} bytes. Use 4, 8, 12, or 16.",
                n
            ),
        }
    }

    pub fn render_integer_ty(&self, int_ty: &IntegerTy) -> String {
        match int_ty {
            IntegerTy::U32 => "u32".to_string(),
            IntegerTy::I32 => "i32".to_string(),
        }
    }

    pub fn render_array_inner(&self, inner: &MaybeAtomic<IntegerTy, BasicTyOrStructRef>) -> String {
        match inner {
            MaybeAtomic::Naked(b) => format!("atomic<{}>", self.render_basic_ty_or_struct_ref(b)),
            MaybeAtomic::Atomic(a) => self.render_integer_ty(a),
        }
    }

    pub fn render_basic_ty(&self, b: &BasicTy) -> String {
        match b {
            BasicTy::F32 => "f32".to_string(),
            BasicTy::Bool => "bool".to_string(),
            BasicTy::Integer(int_ty) => self.render_integer_ty(int_ty),
        }
    }

    pub fn render_basic_ty_or_struct_ref(&self, g: &BasicTyOrStructRef) -> String {
        match g {
            BasicTyOrStructRef::BasicTy(b) => self.render_basic_ty(b),
            BasicTyOrStructRef::StructRef { ident } => ident.clone(),
        }
    }

    pub fn render_binded_buffers(&self) -> String {
        self.ir
            .binded
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let var_str = match &b.ty {
                    CustomBufferBindingType::Storage { read_only } => {
                        let rw = if *read_only { "read" } else { "read_write" };
                        format!("var<{rw}>")
                    }
                    CustomBufferBindingType::Uniform => "var<uniform>".to_string(),
                };

                format!(
                    "@group(0) @binding({i}) {var_str} {}: {};\n",
                    b.ident, b.struct_name
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn render_struct(&self, ident: &str, s: &Struct) -> String {
        let f = s
            .inner
            .iter()
            .map(|(name, ty)| format!("\t{name}: {},\n", format!("{}", self.render_dtype(ty))))
            .collect::<Vec<_>>()
            .join("");

        format!("struct {} {{\n{}}}", ident, f)
    }

    pub fn render_structs(&self) -> String {
        self.ir
            .structs
            .iter()
            .map(|(ident, s)| self.render_struct(ident, s))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn render_funcs(&self) -> String {
        self.ir
            .functions
            .iter()
            .map(|f| {
                let args_str = if f.entrypoint_ty.is_none(){f
                    .args
                    .iter()
                    .map(|(n, d)| format!("{}:{}", n, self.render_dtype(d)))
                    .collect::<Vec<_>>()
                    .join(", ")}else{"".to_string()};

                let body_str = self.render_scope(&f.body, 1);
                let ret_str = f
                    .ret
                    .as_ref()
                    .map(|r| format!("->{}", self.render_basic_ty_or_struct_ref(r)))
                    .unwrap_or_default();

                let entrypoint_headers_str = f
                    .entrypoint_ty
                    .as_ref()
                    .map(|e| match e {
                        EntrypointData::Compute { workgroup_sz } => {
                            format!("@compute @workgroup_size({workgroup_sz})\n")
                        }
                        EntrypointData::Shader => todo!(), // Shader requires proper builtin mangement
                    })
                    .unwrap_or_default();

                format!(
                    "{}fn {}({}){}{{\n{}\n}}",
                    entrypoint_headers_str, f.ident, args_str, ret_str, body_str
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn render_ast(&self, curr_scope: &Scope, ast: &LoweredAST, indent: usize) -> String {
        let tab = "\t".repeat(indent);

        match ast {
            LoweredAST::BinaryOp { lhs, rhs, op } => {
                let sign = match op {
                    BinOp::Add => "+",
                    BinOp::BitwiseAnd => "&",
                    BinOp::Div => "/",
                    BinOp::Eq => "==",
                    BinOp::Gt => ">",
                    BinOp::LogicalAnd => "&&",
                    BinOp::Mul => "*",
                    BinOp::Shl => "<<",
                    BinOp::Shr => ">>",
                    BinOp::Sub => "-",
                };

                format!(
                    "{} {} {}",
                    self.render_ast(curr_scope, lhs, 0),
                    sign,
                    self.render_ast(curr_scope, rhs, 0)
                )
            }
            LoweredAST::Break => format!("{tab}break;"),
            LoweredAST::Conditional {
                cond,
                true_block,
                else_block,
            } => {
                let cond = self.render_ast(curr_scope, cond, 0);

                let true_block_scope = &curr_scope.child_scopes[true_block.0];
                let else_block_str = else_block
                    .as_ref()
                    .map(|e| {
                        let else_body = self.render_scope(
                            &curr_scope.child_scopes[e.0].borrow(),
                            indent + 1,
                        );
                        format!(" else {{\n{else_body}\n{tab}}}")
                    })
                    .unwrap_or_default();
                format!(
                    "{tab}if ({cond}) {{\n{}\n{tab}}}{else_block_str}",
                    self.render_scope(&true_block_scope.borrow(), indent + 1),
                )
            }
            LoweredAST::Continue => format!("{tab}continue;"),
            LoweredAST::Load(r) => {
                let (ident, index_str) = match r {
                    VarRefType::EntryPointGlobal(b) => (
                        self.ir.entrypoint_globals[b.id].to_string(),
                        b.by.iter()
                            .map(|a| match a {
                                Accessor::Index(expr) => {
                                    format!("[{}]", self.render_ast(curr_scope, expr, 0))
                                }
                                Accessor::Field(name) => format!(".{}", name),
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Local(b) => (
                        curr_scope.local_vars[b.id].name.clone(),
                        b.by.iter()
                            .map(|a| match a {
                                Accessor::Index(expr) => {
                                    format!("[{}]", self.render_ast(curr_scope, expr, 0))
                                }
                                Accessor::Field(name) => format!(".{}", name),
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Global(b) => (
                        self.ir.binded[b.id].ident.clone(),
                        b.by.iter()
                            .map(|a| match a {
                                Accessor::Index(expr) => {
                                    format!("[{}]", self.render_ast(curr_scope, expr, 0))
                                }
                                Accessor::Field(name) => format!(".{}", name),
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Shared(b) => (
                        self.ir.shared_vars[b.id].0.clone(),
                        b.by.iter()
                            .map(|a| match a {
                                Accessor::Index(expr) => {
                                    format!("[{}]", self.render_ast(curr_scope, expr, 0))
                                }
                                Accessor::Field(name) => format!(".{}", name),
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                };

                format!("{ident}{index_str}")
            }
            LoweredAST::ForLoop {
                init,
                halt_cond,
                increment,
                body,
            } => {
                let init_str = init.as_ref().map(|i| self.render_ast(curr_scope, i, 0));
                let halt_cond_str = halt_cond.as_ref().map(|h| self.render_ast(curr_scope, h, 0));
                let increment_str = increment.as_ref().map(|i| self.render_ast(curr_scope, i, 0));

                let cond_block = [init_str, halt_cond_str, increment_str]
                    .into_iter()
                    .filter_map(|f| f)
                    .collect::<Vec<_>>()
                    .join("; ");
                let body_str = self.render_scope(&curr_scope.child_scopes[body.0].borrow(), indent + 1);

                format!("{tab}for ({cond_block}) {{\n{body_str}\n{tab}}}")
            }
            LoweredAST::WhileLoop { cond, body } => {
                let cond_str = self.render_ast(curr_scope, cond, 0);
                let body_str = self.render_scope(&curr_scope.child_scopes[body.0].borrow(), indent + 1);

                format!("{tab}while ({cond_str}) {{\n{body_str}\n{tab}}}")
            }
            LoweredAST::Return => format!("{tab}return;"),
            LoweredAST::UnaryOp { operand, op } => {
                let op_str = match op {
                    UnaryOp::BitwiseNot => "!",
                    UnaryOp::LogicalNot => "!",
                    UnaryOp::Neg => "-",
                };

                format!("{}{}", op_str, self.render_ast(curr_scope, operand, 0))
            }
            LoweredAST::Store { var, val } => {
                let (ident, index_str) = match var {
                    VarRefType::EntryPointGlobal(b) => (
                        self.ir.entrypoint_globals[b.id].to_string(),
                        b.by.iter()
                            .map(|a| match a {
                                Accessor::Index(expr) => {
                                    format!("[{}]", self.render_ast(curr_scope, expr, 0))
                                }
                                Accessor::Field(name) => format!(".{}", name),
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Local(b) => (
                        curr_scope.local_vars[b.id].name.clone(),
                        b.by.iter()
                            .map(|a| match a {
                                Accessor::Index(expr) => {
                                    format!("[{}]", self.render_ast(curr_scope, expr, 0))
                                }
                                Accessor::Field(name) => format!(".{}", name),
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Global(b) => (
                        self.ir.binded[b.id].ident.clone(),
                        b.by.iter()
                            .map(|a| match a {
                                Accessor::Index(expr) => {
                                    format!("[{}]", self.render_ast(curr_scope, expr, 0))
                                }
                                Accessor::Field(name) => format!(".{}", name),
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Shared(b) => (
                        self.ir.shared_vars[b.id].0.clone(),
                        b.by.iter()
                            .map(|a| match a {
                                Accessor::Index(expr) => {
                                    format!("[{}]", self.render_ast(curr_scope, expr, 0))
                                }
                                Accessor::Field(name) => format!(".{}", name),
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                };

                format!("{tab}{ident}{index_str} = {};", self.render_ast(curr_scope, val, 0))
            }
            LoweredAST::Const { dt, data } => {
                let (s, _) = match dt {
                    DType::Atomic(_) => panic!("Cannot have atomic constants"),
                    DType::Basic(b) => self.render_basic_ty_const(b, data, 0),
                    DType::Vector(v) => self.render_vec_const(v, data, 0),
                    DType::StructRef { ident } => {
                        let s = self.ir.structs.get(ident).unwrap();
                        self.render_struct_const(ident, s, data, 0)
                    }
                    DType::Pad(bytes) => self.render_pad_const(*bytes, data, 0),
                };
                s
            }
            LoweredAST::FunctionCall { ident, args } => format!(
                "{}({})",
                ident,
                args.iter()
                    .map(|a| self.render_ast(curr_scope, a, 0))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            LoweredAST::Group(stmts) => stmts
                .iter()
                .map(|s| {
                    let r = self.render_ast(curr_scope, s, indent);
                    if r.ends_with('}') || r.ends_with(';') {
                        r
                    } else {
                        r + ";"
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    fn render_basic_ty_const(&self, b: &BasicTy, data: &[u8], offset: usize) -> (String, usize) {
        match b {
            BasicTy::F32 => {
                let bytes = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                if val.is_infinite() {
                    if val.is_sign_negative() {
                        (String::from("-0x1p+127f"), 4)
                    } else {
                        (String::from("0x1p+127f"), 4)
                    }
                } else {
                    (format!("{}f", val), 4)
                }
            }
            BasicTy::Bool => {
                let bytes = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let val = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                (
                    if val != 0 {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    },
                    4,
                )
            }
            BasicTy::Integer(int_ty) => match int_ty {
                IntegerTy::I32 => {
                    let bytes = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                    let val = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    (format!("{}i", val), 4)
                }
                IntegerTy::U32 => {
                    let bytes = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                    let val = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    (format!("{}u", val), 4)
                }
            },
        }
    }

    fn render_struct_const(
        &self,
        ident: &str,
        s: &Struct,
        data: &[u8],
        mut offset: usize,
    ) -> (String, usize) {
        let mut field_inits = Vec::new();
        let mut total_consumed = 0;

        for (field_name, field_ty) in &s.inner {
            let (field_val, consumed) = match field_ty {
                DType::Atomic(_) => panic!("Cannot have atomic constants"),
                DType::Basic(b) => self.render_basic_ty_const(b, data, offset),
                DType::Vector(v) => self.render_vec_const(v, data, offset),
                DType::StructRef { ident } => {
                    let nested = self
                        .ir
                        .structs
                        .get(ident)
                        .unwrap_or_else(|| panic!("Nested struct {} not found", ident));
                    self.render_struct_const(ident, nested, data, offset)
                }
                DType::Pad(bytes) => self.render_pad_const(*bytes, data, offset),
            };
            field_inits.push(format!("{field_name}: {field_val}"));
            offset += consumed;
            total_consumed += consumed;
        }

        let init = format!("{}({})", ident, field_inits.join(", "));
        (init, total_consumed)
    }

    fn render_vec_const(&self, v: &VecTy, data: &[u8], offset: usize) -> (String, usize) {
        match v {
            VecTy::Vec2(inner) => {
                let inner_ty = self.render_basic_ty(inner);
                let (val0, off0) = self.render_basic_ty_const(inner, data, offset);
                let (val1, off1) = self.render_basic_ty_const(inner, data, offset + off0);
                (format!("vec2<{inner_ty}>({val0}, {val1})"), off0 + off1)
            }
            VecTy::Vec3(inner) => {
                let inner_ty = self.render_basic_ty(inner);
                let (val0, off0) = self.render_basic_ty_const(inner, data, offset);
                let (val1, off1) = self.render_basic_ty_const(inner, data, offset + off0);
                let (val2, off2) = self.render_basic_ty_const(inner, data, offset + off0 + off1);
                (
                    format!("vec3<{inner_ty}>({val0}, {val1}, {val2})"),
                    off0 + off1 + off2,
                )
            }
            VecTy::Vec4(inner) => {
                let inner_ty = self.render_basic_ty(inner);
                let (val0, off0) = self.render_basic_ty_const(inner, data, offset);
                let (val1, off1) = self.render_basic_ty_const(inner, data, offset + off0);
                let (val2, off2) = self.render_basic_ty_const(inner, data, offset + off0 + off1);
                let (val3, off3) =
                    self.render_basic_ty_const(inner, data, offset + off0 + off1 + off2);
                (
                    format!("vec4<{inner_ty}>({val0}, {val1}, {val2}, {val3})"),
                    off0 + off1 + off2 + off3,
                )
            }
            VecTy::Array(_inner) => {
                unimplemented!("Can be lowered further with for loops and shit so ye")
            }
        }
    }

    fn render_pad_const(&self, bytes: usize, data: &[u8], offset: usize) -> (String, usize) {
        let val = match bytes {
            4 => {
                let b = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let v0 = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                format!("{}u", v0)
            }
            8 => {
                let b0 = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let b1 = data.get(offset + 4..offset + 8).unwrap_or(&[0; 4]);
                let v0 = u32::from_le_bytes([b0[0], b0[1], b0[2], b0[3]]);
                let v1 = u32::from_le_bytes([b1[0], b1[1], b1[2], b1[3]]);
                format!("vec2<u32>({}u, {}u)", v0, v1)
            }
            12 => {
                let b0 = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let b1 = data.get(offset + 4..offset + 8).unwrap_or(&[0; 4]);
                let b2 = data.get(offset + 8..offset + 12).unwrap_or(&[0; 4]);
                let v0 = u32::from_le_bytes([b0[0], b0[1], b0[2], b0[3]]);
                let v1 = u32::from_le_bytes([b1[0], b1[1], b1[2], b1[3]]);
                let v2 = u32::from_le_bytes([b2[0], b2[1], b2[2], b2[3]]);
                format!("vec3<u32>({}u, {}u, {}u)", v0, v1, v2)
            }
            16 => {
                let b0 = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let b1 = data.get(offset + 4..offset + 8).unwrap_or(&[0; 4]);
                let b2 = data.get(offset + 8..offset + 12).unwrap_or(&[0; 4]);
                let b3 = data.get(offset + 12..offset + 16).unwrap_or(&[0; 4]);
                let v0 = u32::from_le_bytes([b0[0], b0[1], b0[2], b0[3]]);
                let v1 = u32::from_le_bytes([b1[0], b1[1], b1[2], b1[3]]);
                let v2 = u32::from_le_bytes([b2[0], b2[1], b2[2], b2[3]]);
                let v3 = u32::from_le_bytes([b3[0], b3[1], b3[2], b3[3]]);
                format!("vec4<u32>({}u, {}u, {}u, {}u)", v0, v1, v2, v3)
            }
            n => panic!("Unsupported padding size: {} bytes", n),
        };
        (val, bytes)
    }

    pub fn render_workgroup_vars(&self) -> String {
        self.ir
            .shared_vars
            .iter()
            .map(|(name, dt)| format!("var<workgroup> {}: {};\n", name, self.render_dtype(dt)))
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn render_scope(&self, scope: &Scope, indent: usize) -> String {
        let tab = "\t".repeat(indent);
        let decls: String = scope
            .local_vars
            .iter()
            .skip(scope.num_inherited_locals)
            .map(|v| {
                let kw = if v.mut_ { "var" } else { "let" };
                format!(
                    "{tab}{} {}: {} = {};\n",
                    kw,
                    v.name,
                    self.render_dtype(&v.ast.dt(&self.ir, scope)),
                    self.render_ast(scope, &v.ast, 0),
                )
            })
            .collect();
        let body = self.render_ast(scope, scope.ast.as_ref().unwrap(), indent);
        format!("{}{}", decls, body)
    }

    pub fn translate(&self) -> String {
        let structs_str = self.render_structs();
        let bindings_str = self.render_binded_buffers();
        let workgroup_str = self.render_workgroup_vars();

        let funcs_str = self.render_funcs();

        [structs_str, bindings_str, workgroup_str, funcs_str].join("\n\n\n\n")
    }
}
