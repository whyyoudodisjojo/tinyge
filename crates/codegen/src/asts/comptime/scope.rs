use std::{cell::RefCell, rc::Rc};

use crate::asts::comptime::ScopePtr;

use super::{BindedBuffer, ComptimeAST, EntrypointGlobals, ShaderIR};

#[derive(Clone)]
pub struct LocalVariables {
    pub ast: ComptimeAST,
}

pub struct Scope<'a> {
    pub ast: Option<ComptimeAST>,
    pub binded: &'a [BindedBuffer],
    pub entrypoint_globals: &'a [EntrypointGlobals],

    pub local_vars: Vec<LocalVariables>,
    pub child_scopes: Vec<Rc<RefCell<Self>>>,
}

impl<'a> Scope<'a> {
    pub fn new(ir: &'a ShaderIR) -> Self {
        Self {
            ast: None,
            binded: &ir.binded,
            entrypoint_globals: &ir.entrypoint_globals,
            local_vars: vec![],
            child_scopes: vec![],
        }
    }

    pub fn new_scope(&mut self) -> Rc<RefCell<Scope<'a>>> {
        let new = Rc::new(RefCell::new(Self {
            ast: None,
            binded: self.binded,
            entrypoint_globals: self.entrypoint_globals,
            local_vars: self.local_vars.clone(),
            child_scopes: vec![],
        }));
        self.child_scopes.push(new.clone());

        new
    }

    pub fn cond(
        &mut self,
        cond: ComptimeAST,
        then: impl FnOnce(&mut Self),
        el: Option<impl FnOnce(&mut Self)>,
    ) -> ScopePtr {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        then(&mut new_scope);
        let ast = ComptimeAST::Conditional {
            cond: Box::new(cond),
            true_block: new_scope_ptr.clone(),
            else_block: el.map(|e| {
                e(&mut new_scope);
                new_scope_ptr.clone()
            }),
        };

        new_scope.ast = Some(ast);
        new_scope_ptr
    }

    pub fn while_loop(&mut self, cond: ComptimeAST, body: impl FnOnce(&mut Self)) -> ScopePtr {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        body(&mut new_scope);
        let ast = ComptimeAST::WhileLoop {
            cond: Box::new(cond),
            body: new_scope_ptr.clone(),
        };

        new_scope.ast = Some(ast);
        new_scope_ptr
    }

    pub fn for_loop(
        &mut self,
        init: Option<ComptimeAST>,
        halt_cond: Option<ComptimeAST>,
        increment: Option<ComptimeAST>,
        body: impl FnOnce(&mut Self),
    ) -> ScopePtr {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        body(&mut new_scope);
        let ast = ComptimeAST::ForLoop {
            init: init.map(|i| Box::new(i)),
            halt_cond: halt_cond.map(|h| Box::new(h)),
            increment: increment.map(|increment| Box::new(increment)),
            body: new_scope_ptr.clone(),
        };

        new_scope.ast = Some(ast);
        new_scope_ptr
    }
}
