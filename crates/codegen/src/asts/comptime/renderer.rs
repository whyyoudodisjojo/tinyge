use wgpu::BufferBindingType;

use crate::{
    asts::comptime::{
        BinOp,
        ComptimeAST::{self},
        ShaderIR, Struct, UnaryOp, VarRefType,
        scope::Scope,
    },
    dt::{BasicTy, BasicTyOrStructRef, BasicTyOrStructRefOrStructDef, DType, MaybeAtomic, VecTy},
};

pub struct WgslRenderer<'a> {
    ir: &'a ShaderIR<'a>,
}

impl<'a> WgslRenderer<'a> {
    pub fn render_dtype(&self, dt: &DType) -> String {
        match dt {
            DType::MaybeAtomic(m_a) => format!("{}", self.render_maybe_atomics_ref(m_a)),
            DType::Vector(v) => match v {
                VecTy::Array(a) => format!("array<{}>", self.render_maybe_atomics_ref(a)),
                VecTy::Vec2(a) => format!("vec2<{}>", self.render_maybe_atomics_ref(a)),
                VecTy::Vec3(a) => format!("vec3<{}>", self.render_maybe_atomics_ref(a)),
            },
        }
    }

    pub fn render_dtype_with_struct_ref(&self, dt: &DType<BasicTyOrStructRef>) -> String {
        match dt {
            DType::MaybeAtomic(m_a) => format!("{}", self.render_maybe_atomics_ref(m_a)),
            DType::Vector(v) => match v {
                VecTy::Array(a) => format!("array<{}>", self.render_maybe_atomics_ref(a)),
                VecTy::Vec2(a) => format!("vec2<{}>", self.render_maybe_atomics_ref(a)),
                VecTy::Vec3(a) => format!("vec3<{}>", self.render_maybe_atomics_ref(a)),
            },
        }
    }

    pub fn render_maybe_atomics_def_ref(
        &self,
        m_a: &MaybeAtomic<BasicTyOrStructRefOrStructDef>,
    ) -> String {
        match m_a {
            MaybeAtomic::Atomic(a) => {
                format!("atomic<{}>", self.render_basic_ty_or_struct_ref_or_def(a))
            }
            MaybeAtomic::Naked(n) => format!("{}", self.render_basic_ty_or_struct_ref_or_def(n)),
        }
    }

    pub fn render_maybe_atomics_ref(&self, m_a: &MaybeAtomic<BasicTyOrStructRef>) -> String {
        match m_a {
            MaybeAtomic::Atomic(a) => format!("atomic<{}>", self.render_basic_ty_or_struct_ref(a)),
            MaybeAtomic::Naked(n) => format!("{}", self.render_basic_ty_or_struct_ref(n)),
        }
    }

    pub fn render_basic_ty_or_struct_ref(&self, g: &BasicTyOrStructRef) -> String {
        match g {
            BasicTyOrStructRef::BasicTy(b) => match b {
                BasicTy::F32 => "f32".to_string(),
                BasicTy::I32 => "i32".to_string(),
                BasicTy::U32 => "u32".to_string(),
            },
            BasicTyOrStructRef::StructRef { id } => {
                format!("{}", self.ir.structs[*id].ident)
            }
        }
    }

    pub fn render_basic_ty_or_struct_ref_or_def(
        &self,
        g: &BasicTyOrStructRefOrStructDef,
    ) -> String {
        match g {
            BasicTyOrStructRefOrStructDef::BasicTy(b) => match b {
                BasicTy::F32 => "f32".to_string(),
                BasicTy::I32 => "i32".to_string(),
                BasicTy::U32 => "u32".to_string(),
            },
            BasicTyOrStructRefOrStructDef::StructRef { id } => {
                format!("{}", self.ir.structs[*id].ident)
            }
            BasicTyOrStructRefOrStructDef::StructDef(s) => self.render_struct(s),
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

                let dt = self.render_dtype_with_struct_ref(&b.dt);

                format!("@group(0) @binding({i}) {var_str} {}: {dt}", b.ident)
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn render_struct(&self, s: &Struct) -> String {
        let f = s
            .inner
            .iter()
            .map(|(name, ty)| format!("{name}: {},", format!("{}", self.render_dtype(ty))))
            .collect::<Vec<_>>()
            .join("\n\n");

        format!("struct {} {{{}}}", s.ident, f)
    }

    pub fn render_structs(&self) -> String {
        self.ir
            .structs
            .iter()
            .map(|s| self.render_struct(s))
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
                    .map(|(n, d)| format!("{}:{}", n, self.render_dtype_with_struct_ref(d)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("fn {}({})", f.ident, args_str,)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn render_ast(&self, curr_scope: &Scope, ast: &ComptimeAST) -> String {
        match ast {
            ComptimeAST::BinaryOp { lhs, rhs, op } => {
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
            ComptimeAST::Break => "break;".to_string(),
            ComptimeAST::Conditional {
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
            ComptimeAST::Continue => "continue;".to_string(),
            ComptimeAST::Load(r) => {
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
            ComptimeAST::ForLoop {
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
            ComptimeAST::WhileLoop { cond, body } => {
                let cond_str = self.render_ast(curr_scope, cond);
                let body_str = self.render_scope(&curr_scope.child_scopes[body.0].borrow());

                format!("for({cond_str}){{{body_str}}}")
            }
            ComptimeAST::Return => "return;".to_string(),
            ComptimeAST::UnaryOp { operand, op } => {
                // TODO: Infer atomics and inject atomicops fn calls
                let op_str = match op {
                    UnaryOp::BitwiseNot => "!",
                    UnaryOp::LogicalNot => "!",
                    UnaryOp::Neg => "-",
                };

                format!("{}{}", op_str, self.render_ast(curr_scope, operand))
            }
            ComptimeAST::Store { var, val } => {
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
            ComptimeAST::Const { dt, data } => {
                let (s, _) = match dt {
                    DType::MaybeAtomic(m_a) => self.render_maybe_atomic_const(m_a, data, 0),
                    DType::Vector(v) => self.render_vec_const(v, data, 0),
                };
                s
            }
        }
    }

    fn render_maybe_atomic_const(
        &self,
        m_a: &MaybeAtomic<BasicTyOrStructRef>,
        data: &[u8],
        offset: usize,
    ) -> (String, usize) {
        match m_a {
            MaybeAtomic::Naked(inner) => {
                self.render_basic_ty_or_struct_ref_const(inner, data, offset)
            }
            MaybeAtomic::Atomic(_) => panic!("Cannot have atomic constants"),
        }
    }

    fn render_basic_ty_or_struct_ref_const(
        &self,
        ty: &BasicTyOrStructRef,
        data: &[u8],
        offset: usize,
    ) -> (String, usize) {
        match ty {
            BasicTyOrStructRef::BasicTy(b) => self.render_basic_ty_const(b, data, offset),
            BasicTyOrStructRef::StructRef { id } => {
                let struct_def = &self.ir.structs[*id];
                self.render_struct_const(struct_def, data, offset)
            }
        }
    }

    fn render_basic_ty_const(&self, b: &BasicTy, data: &[u8], offset: usize) -> (String, usize) {
        match b {
            BasicTy::F32 => {
                let bytes = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                (format!("{}f", val), 4)
            }
            BasicTy::I32 => {
                let bytes = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let val = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                (format!("{}i", val), 4)
            }
            BasicTy::U32 => {
                let bytes = data.get(offset..offset + 4).unwrap_or(&[0; 4]);
                let val = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                (format!("{}u", val), 4)
            }
        }
    }

    fn render_struct_const(&self, s: &Struct, data: &[u8], mut offset: usize) -> (String, usize) {
        let mut field_inits = Vec::new();
        let mut total_consumed = 0;

        for (field_name, field_ty) in &s.inner {
            let (field_val, consumed) = match field_ty {
                DType::MaybeAtomic(m_a) => self.render_maybe_atomic_const(m_a, data, offset),
                DType::Vector(v) => self.render_vec_const(v, data, offset),
            };
            field_inits.push(format!("{field_name}: {field_val}"));
            offset += consumed;
            total_consumed += consumed;
        }

        let init = format!("{}({})", s.ident, field_inits.join(", "));
        (init, total_consumed)
    }

    fn render_vec_const(
        &self,
        v: &VecTy<BasicTyOrStructRef>,
        data: &[u8],
        offset: usize,
    ) -> (String, usize) {
        match v {
            VecTy::Vec2(inner) => {
                let inner_ty = self.render_maybe_atomics_ref(inner);
                let (val0, off0) = self.render_maybe_atomic_const_scalar(inner, data, offset);
                let (val1, off1) =
                    self.render_maybe_atomic_const_scalar(inner, data, offset + off0);
                (format!("vec2<{inner_ty}>({val0}, {val1})"), off0 + off1)
            }
            VecTy::Vec3(inner) => {
                let inner_ty = self.render_maybe_atomics_ref(inner);
                let (val0, off0) = self.render_maybe_atomic_const_scalar(inner, data, offset);
                let (val1, off1) =
                    self.render_maybe_atomic_const_scalar(inner, data, offset + off0);
                let (val2, off2) =
                    self.render_maybe_atomic_const_scalar(inner, data, offset + off0 + off1);
                (
                    format!("vec3<{inner_ty}>({val0}, {val1}, {val2})"),
                    off0 + off1 + off2,
                )
            }
            VecTy::Array(_inner) => {
                todo!("array constants not yet implemented")
            }
        }
    }

    fn render_maybe_atomic_const_scalar(
        &self,
        m_a: &MaybeAtomic<BasicTyOrStructRef>,
        data: &[u8],
        offset: usize,
    ) -> (String, usize) {
        match m_a {
            MaybeAtomic::Naked(inner) => match inner {
                BasicTyOrStructRef::BasicTy(b) => self.render_basic_ty_const(b, data, offset),
                BasicTyOrStructRef::StructRef { .. } => {
                    panic!("Cannot use struct type as vector element")
                }
            },
            MaybeAtomic::Atomic(_) => todo!("cannot have atomic constants, without atomicstore and all"),
        }
    }

    pub fn render_scope(&self, scope: &Scope) -> String {
        todo!()
    }

    pub fn translate(&self) -> String {
        let structs_str = self.render_structs();
        let bindings_str = self.render_binded_buffers();

        todo!()
    }
}
