import numpy as np
import re


class TreeNode:
    def __init__(self, val: str, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right



class ExpressionTree:
    root = None
    operators = {'+': 0, '-': 0, '*': 1, '/': 1, '^': 2}

    operations = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        '^': lambda x, y: x**y
    }

    def __init__(self, expression=None, root=None):
        if expression is not None:
            tokens = self.tokenize(expression)
            postfix = self.postfix(tokens)
            self.build_tree(postfix)
        if root is not None:
            self.root = root

    @staticmethod
    def tokenize(expression: str):
        """
        Tokenize a string expression
        :param expression: Expression to tokenize
        :return: List of tokens
        """
        token_pattern = r'[a-zA-Z_][a-zA-Z_0-9]*|\d+|[+\-*/^()x]'
        return re.findall(token_pattern, expression)

    @staticmethod
    def postfix(tokens: list):
        """
        Given a list of tokens in infix notation, convert them to postfix notation
        :param tokens: Infix notation tokens
        :return: Postfix notation tokens
        """
        tokens.append(')')
        postfix_expression = []
        stack = ['(']
        for token in tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                while True:
                    top = stack.pop(-1)
                    if top != '(':
                        postfix_expression.append(top)
                    else:
                        break
            elif token in ExpressionTree.operators:
                while True:
                    top = stack[-1]
                    if top == '(' or ExpressionTree.operators[top] - ExpressionTree.operators[token] < 0:
                        stack.append(token)
                        break
                    else:
                        postfix_expression.append(top)
                        stack = stack[:-1]
            else:
                postfix_expression.append(token)
        return postfix_expression

    def build_tree(self, tokens):
        """
        Builds the tree given a list of tokens in postfix notation
        :param tokens: List of tokens
        """
        stack = []
        for token in tokens:
            if token not in ExpressionTree.operators:
                stack.append(TreeNode(token))
            else:
                cur_node = TreeNode(token)
                cur_node.right = stack.pop(-1)
                cur_node.left = stack.pop(-1)
                stack.append(cur_node)

        self.root = stack.pop(-1)

    def __call__(self, vars: dict):
        """
        Evaluates the tree given specified variable values
        :param vars: Variable values
        :return: The evaluted expression
        """
        return self.eval(self.root, vars)

    def eval(self, node: TreeNode, vars: dict):
        """
        Recursively evaluates tree nodes
        :param node: Current node
        :param vars: Variable dictionary
        :return: Value at the current node
        """
        if node.val in self.operators:
            left_val = self.eval(node.left, vars)
            right_val = self.eval(node.right, vars)
            return self.operations[node.val](left_val, right_val)
        elif node.val in vars:
            return vars[node.val]
        elif node.val.isnumeric():
            return float(node.val)
        raise Exception("All variables require values")

    def __add__(self, other):
        newRoot = TreeNode('+')
        newRoot.left = self.root
        newRoot.right = other.root
        return ExpressionTree(root=newRoot)

    def __iadd__(self, other):
        newRoot = TreeNode('+')
        newRoot.left = self.root
        newRoot.right = other.root
        self.root = newRoot
        return self





expression1 = f'x+y'
expression2 = f'2*x'

vars = {'x': 5, 'y': 3}

tree1 = ExpressionTree(expression1)
tree2 = ExpressionTree(expression2)
print(tree1(vars))
print(tree2(vars))
print((tree1+tree2)(vars))
tree1 += tree2
print(tree1(vars))
