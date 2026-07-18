use std::{cell::RefCell, rc::Rc};

use crate::asts::lowered::ScopePtr;

use super::LoweredAST;

#[derive(Clone)]
pub struct LocalVariables {
    pub name: String,
    pub mut_: bool,
    pub ast: LoweredAST,
}

pub struct Scope {
    pub ast: Option<LoweredAST>,

    pub local_vars: Vec<LocalVariables>,
    pub num_inherited_locals: usize,
    pub child_scopes: Vec<Rc<RefCell<Self>>>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            ast: None,
            local_vars: vec![],
            num_inherited_locals: 0,
            child_scopes: vec![],
        }
    }

    pub fn add_local(&mut self, name: String, mut_: bool, ast: LoweredAST) -> usize {
        let id = self.local_vars.len();
        self.local_vars.push(LocalVariables { name, mut_, ast });
        id
    }

    pub fn new_scope(&mut self) -> Rc<RefCell<Scope>> {
        let num_inherited = self.local_vars.len();
        let new = Rc::new(RefCell::new(Self {
            ast: None,
            local_vars: self.local_vars.clone(),
            num_inherited_locals: num_inherited,
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
    ) -> LoweredAST {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        then(&mut new_scope);
        LoweredAST::Conditional {
            cond: Box::new(cond),
            true_block: new_scope_ptr.clone(),
            else_block: el.map(|e| {
                e(&mut new_scope);
                new_scope_ptr.clone()
            }),
        }
    }

    pub fn while_loop(&mut self, cond: LoweredAST, body: impl FnOnce(&mut Self)) -> LoweredAST {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        body(&mut new_scope);
        LoweredAST::WhileLoop {
            cond: Box::new(cond),
            body: new_scope_ptr,
        }
    }

    pub fn for_loop(
        &mut self,
        init: Option<LoweredAST>,
        halt_cond: Option<LoweredAST>,
        increment: Option<LoweredAST>,
        body: impl FnOnce(&mut Self),
    ) -> LoweredAST {
        let new_scope = self.new_scope();
        let mut new_scope = new_scope.borrow_mut();
        let new_scope_ptr = ScopePtr(self.child_scopes.len() - 1);
        body(&mut new_scope);
        LoweredAST::ForLoop {
            init: init.map(|i| Box::new(i)),
            halt_cond: halt_cond.map(|h| Box::new(h)),
            increment: increment.map(|increment| Box::new(increment)),
            body: new_scope_ptr,
        }
    }
}
