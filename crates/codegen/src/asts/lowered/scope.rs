use std::{cell::RefCell, rc::Rc};

use crate::asts::lowered::ScopePtr;

use super::LoweredAST;

#[derive(Clone)]
pub struct LocalVariables {
    pub ast: LoweredAST,
}

pub struct Scope {
    pub ast: Option<LoweredAST>,

    pub local_vars: Vec<LocalVariables>,
    pub child_scopes: Vec<Rc<RefCell<Self>>>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            ast: None,
            local_vars: vec![],
            child_scopes: vec![],
        }
    }

    pub fn new_scope(&mut self) -> Rc<RefCell<Scope>> {
        let new = Rc::new(RefCell::new(Self {
            ast: None,
            local_vars: self.local_vars.clone(),
            child_scopes: vec![],
        }));
        self.child_scopes.push(new.clone());

        new
    }

    pub fn cond(
        &mut self,
        cond: LoweredAST,
        then: impl FnOnce(&mut Self),
        el: Option<impl FnOnce(&mut Self)>,
    ) -> ScopePtr {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        then(&mut new_scope);
        let ast = LoweredAST::Conditional {
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

    pub fn while_loop(&mut self, cond: LoweredAST, body: impl FnOnce(&mut Self)) -> ScopePtr {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        body(&mut new_scope);
        let ast = LoweredAST::WhileLoop {
            cond: Box::new(cond),
            body: new_scope_ptr.clone(),
        };

        new_scope.ast = Some(ast);
        new_scope_ptr
    }

    pub fn for_loop(
        &mut self,
        init: Option<LoweredAST>,
        halt_cond: Option<LoweredAST>,
        increment: Option<LoweredAST>,
        body: impl FnOnce(&mut Self),
    ) -> ScopePtr {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        body(&mut new_scope);
        let ast = LoweredAST::ForLoop {
            init: init.map(|i| Box::new(i)),
            halt_cond: halt_cond.map(|h| Box::new(h)),
            increment: increment.map(|increment| Box::new(increment)),
            body: new_scope_ptr.clone(),
        };

        new_scope.ast = Some(ast);
        new_scope_ptr
    }
}
