use wgpu::BufferBindingType;

use crate::{
    asts::lowered::{
        BinOp, EntrypointData,
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
    ir: &'a ShaderIR<'a>,
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
            },
            DType::StructRef { ident } => ident.clone(),
        }
    }

    pub fn render_integer_ty(&self, int_ty: &IntegerTy) -> String {
        match int_ty {
            IntegerTy::U32 => "u32".to_string(),
            IntegerTy::I32 => "i32".to_string(),
        }
    }

    pub fn render_array_inner(&self, inner: &MaybeAtomic<BasicTy>) -> String {
        match inner {
            MaybeAtomic::Atomic(b) => format!("atomic<{}>", self.render_basic_ty(b)),
            MaybeAtomic::Naked(b) => self.render_basic_ty(b),
        }
    }

    pub fn render_basic_ty(&self, b: &BasicTy) -> String {
        match b {
            BasicTy::F32 => "f32".to_string(),
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
                    BufferBindingType::Storage { read_only } => {
                        let rw = if *read_only { "read" } else { "read_write" };
                        format!("var<{rw}>")
                    }
                    BufferBindingType::Uniform => "var<uniform>".to_string(),
                };

                let dt = self.render_dtype(&b.dt);

                format!("@group(0) @binding({i}) {var_str} {}: {dt}", b.ident)
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn render_struct(&self, ident: &str, s: &Struct) -> String {
        let f = s
            .inner
            .iter()
            .map(|(name, ty)| format!("{name}: {},", format!("{}", self.render_dtype(ty))))
            .collect::<Vec<_>>()
            .join("\n\n");

        format!("struct {} {{{}}}", ident, f)
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
                let args_str = f
                    .args
                    .iter()
                    .map(|(n, d)| format!("{}:{}", n, self.render_dtype(d)))
                    .collect::<Vec<_>>()
                    .join(", ");

                let body_str = self.render_scope(&f.body);
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
                            format!("@compute @workgroup_size{workgroup_sz}")
                        }
                        EntrypointData::Shader => todo!(), // Shader requires proper builtin mangement
                    })
                    .unwrap_or_default();

                format!(
                    "{}fn {}({}){}{{{}}}",
                    entrypoint_headers_str, f.ident, args_str, ret_str, body_str
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn render_ast(&self, curr_scope: &Scope, ast: &LoweredAST) -> String {
        match ast {
            LoweredAST::BinaryOp { lhs, rhs, op } => {
                // TODO: Infer atomics and add atomicops fn call
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
                    "{}{}{}",
                    self.render_ast(curr_scope, lhs),
                    sign,
                    self.render_ast(curr_scope, rhs)
                )
            }
            LoweredAST::Break => "break;".to_string(),
            LoweredAST::Conditional {
                cond,
                true_block,
                else_block,
            } => {
                let cond = self.render_ast(curr_scope, cond);

                let true_block_scope = &curr_scope.child_scopes[true_block.0];
                let else_block_str = else_block
                    .as_ref()
                    .map(|e| {
                        format!(
                            "else{{{}}}",
                            self.render_scope(&curr_scope.child_scopes[e.0].borrow())
                        )
                    })
                    .unwrap_or_default();
                format!(
                    "if ({}){{{}}}{}",
                    cond,
                    self.render_scope(&true_block_scope.borrow()),
                    else_block_str
                )
            }
            LoweredAST::Continue => "continue;".to_string(),
            LoweredAST::Load(r) => {
                // TODO: infer atomics and use atomicload
                let (ident, index_str) = match r {
                    VarRefType::EntryPointGlobal(b) => (
                        curr_scope.entrypoint_globals[b.id].to_string(),
                        b.by_index
                            .iter()
                            .map(|i| format!("[{i}]"))
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Local(b) => (
                        self.render_ast(curr_scope, &curr_scope.local_vars[b.id].ast),
                        b.by_index
                            .iter()
                            .map(|i| format!("[{i}]"))
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Global(b) => (
                        curr_scope.binded[b.id].ident.clone(),
                        b.by_index
                            .iter()
                            .map(|i| format!("[{i}]"))
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
                let init_str = init.as_ref().map(|i| self.render_ast(curr_scope, &i));
                let halt_cond_str = halt_cond.as_ref().map(|h| self.render_ast(curr_scope, &h));
                let increment_str = increment.as_ref().map(|i| self.render_ast(curr_scope, &i));

                let cond_block = [init_str, halt_cond_str, increment_str]
                    .into_iter()
                    .filter_map(|f| f)
                    .collect::<Vec<_>>()
                    .join("; ");
                let body_str = self.render_scope(&curr_scope.child_scopes[body.0].borrow());

                format!("for({cond_block}){{{body_str}}}")
            }
            LoweredAST::WhileLoop { cond, body } => {
                let cond_str = self.render_ast(curr_scope, cond);
                let body_str = self.render_scope(&curr_scope.child_scopes[body.0].borrow());

                format!("for({cond_str}){{{body_str}}}")
            }
            LoweredAST::Return => "return;".to_string(),
            LoweredAST::UnaryOp { operand, op } => {
                // TODO: Infer atomics and inject atomicops fn calls
                let op_str = match op {
                    UnaryOp::BitwiseNot => "!",
                    UnaryOp::LogicalNot => "!",
                    UnaryOp::Neg => "-",
                };

                format!("{}{}", op_str, self.render_ast(curr_scope, operand))
            }
            LoweredAST::Store { var, val } => {
                // TODO: Infer atomics and inject atomicstore
                let (ident, index_str) = match var {
                    VarRefType::EntryPointGlobal(b) => (
                        curr_scope.entrypoint_globals[b.id].to_string(),
                        b.by_index
                            .iter()
                            .map(|i| format!("[{i}]"))
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Local(b) => (
                        self.render_ast(curr_scope, &curr_scope.local_vars[b.id].ast),
                        b.by_index
                            .iter()
                            .map(|i| format!("[{i}]"))
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                    VarRefType::Global(b) => (
                        curr_scope.binded[b.id].ident.clone(),
                        b.by_index
                            .iter()
                            .map(|i| format!("[{i}]"))
                            .collect::<Vec<_>>()
                            .join(""),
                    ),
                };

                format!("{ident}{index_str} = {};", self.render_ast(curr_scope, val))
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
                };
                s
            }
            LoweredAST::FunctionCall { ident, args } => format!(
                "{ident}({});",
                args.iter()
                    .map(|a| self.render_ast(curr_scope, a))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    fn render_basic_ty_const(&self, b: &BasicTy, data: &[u8], offset: usize) -> (String, usize) {
        match b {
            BasicTy::F32 => {
                let bytes = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                (format!("{}f", val), 4)
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
            VecTy::Array(_inner) => {
                unimplemented!("Can be lowered further with for loops and shit so ye")
            }
        }
    }

    pub fn render_scope(&self, scope: &Scope) -> String {
        self.render_ast(scope, scope.ast.as_ref().unwrap())
    }

    pub fn translate(&self) -> String {
        let structs_str = self.render_structs();
        let bindings_str = self.render_binded_buffers();

        let funcs_str = self.render_funcs();

        [structs_str, bindings_str, funcs_str].join("\n\n\n\n")
    }
}
