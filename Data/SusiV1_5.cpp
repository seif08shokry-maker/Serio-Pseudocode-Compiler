#include <iostream>
#include <string>
#include <vector>
#include <cctype>
#include <map>
#include <stdexcept>
#include <memory>
#include <fstream>
#include <algorithm>
#include <variant>
#include "susi_compiler_api.h"


// --- Tokenizer and Lexer Code ---


// Token types
enum class TokenType {
    // Keywords
    OUTPUT,
    INPUT,
    IF,
    ELSE,
    ENDIF,
    DECLARE,
    BOOLEAN,
    INTEGER,
    STRING_TYPE,    // <-- New: STRING data type keyword
    REAL,           // <-- New: REAL data type
    ARRAY1D,        // <-- New: 1D ARRAY data type
    ARRAY2D,        // <-- New: 2D ARRAY data type
    THEN,
    WAIT,
    FOR,            // <-- New: FOR loop
    TO,             // <-- New: TO keyword for FOR loops
    ENDFOR,         // <-- New: ENDFOR to close FOR loops
    WHILE,
    ENDWHILE,
    CASE,
    OF,
    OTHERWISE,
    ENDCASE,
    TYPE,
    ENDTYPE,
    // Literals
    NUMBER,
    REAL_NUMBER,    // <-- New: Real number literal (e.g., 3.14)
    STRING,
    TRUE,
    FALSE,
    YES,
    NO,
    REPEAT,
    UNTIL,
    // Symbols and operators
    ASSIGN,         // :=
    EQUAL,          // =
    PLUS,           // +
    MINUS,          // -
    MULTIPLY,       // *
    DIVIDE,         // /
    MORE,           // >
    LESS,           // <
    MOREOREQUAL,    // >=
    LESSOREQUAL,    // <=
    NOTEQUAL,       // !=
    LEFT_PAREN,     // (
    RIGHT_PAREN,    // )
    LEFT_BRACKET,   // [ <-- New: For array indexing
    RIGHT_BRACKET,  // ] <-- New: For array indexing
    COMMA,          // , <-- New: For array dimensions and initialization
    COLON,
    DOT,
    // Others
    IDENTIFIER,
    ENDOFFILE
};


// Token with location
struct Token {
    TokenType type;
    std::string lexeme;
    int line = 1;
    int col = 1;
};


// Helper: token type to string (for diagnostics)
std::string tokenTypeToString(TokenType type) {
    switch (type) {
        case TokenType::OUTPUT: return "OUTPUT";
        case TokenType::INPUT: return "INPUT";
        case TokenType::IF: return "IF";
        case TokenType::ELSE: return "ELSE";
        case TokenType::ENDIF: return "ENDIF";
        case TokenType::DECLARE: return "DECLARE";
        case TokenType::BOOLEAN: return "BOOLEAN";
        case TokenType::INTEGER: return "INTEGER";
        case TokenType::STRING_TYPE: return "STRING_TYPE";
        case TokenType::REAL: return "REAL";
        case TokenType::ARRAY1D: return "ARRAY1D";
        case TokenType::ARRAY2D: return "ARRAY2D";
        case TokenType::THEN: return "THEN";
        case TokenType::WAIT: return "WAIT";
        case TokenType::FOR: return "FOR";
        case TokenType::TO: return "TO";
        case TokenType::ENDFOR: return "ENDFOR";
        case TokenType::NUMBER: return "NUMBER";
        case TokenType::REAL_NUMBER: return "REAL_NUMBER";
        case TokenType::STRING: return "STRING";
        case TokenType::TRUE: return "TRUE";
        case TokenType::FALSE: return "FALSE";
        case TokenType::YES: return "YES";
        case TokenType::NO: return "NO";
        case TokenType::ASSIGN: return "ASSIGN";
        case TokenType::EQUAL: return "EQUAL";
        case TokenType::PLUS: return "PLUS";
        case TokenType::MINUS: return "MINUS";
        case TokenType::MULTIPLY: return "MULTIPLY";
        case TokenType::DIVIDE: return "DIVIDE";
        case TokenType::MORE: return "MORE";
        case TokenType::LESS: return "LESS";
        case TokenType::MOREOREQUAL: return "MOREOREQUAL";
        case TokenType::LESSOREQUAL: return "LESSOREQUAL";
        case TokenType::NOTEQUAL: return "NOTEQUAL";
        case TokenType::LEFT_PAREN: return "LEFT_PAREN";
        case TokenType::RIGHT_PAREN: return "RIGHT_PAREN";
        case TokenType::LEFT_BRACKET: return "LEFT_BRACKET";
        case TokenType::RIGHT_BRACKET: return "RIGHT_BRACKET";
        case TokenType::COMMA: return "COMMA";
        case TokenType::IDENTIFIER: return "IDENTIFIER";
        case TokenType::ENDOFFILE: return "ENDOFFILE";
        case TokenType::WHILE: return "WHILE";
        case TokenType::ENDWHILE: return "ENDWHILE";
        case TokenType::CASE: return "CASE";
        case TokenType::OF: return "OF";
        case TokenType::OTHERWISE: return "OTHERWISE";
        case TokenType::ENDCASE: return "ENDCASE";
        case TokenType::COLON: return "COLON";
        case TokenType::REPEAT: return "REPEAT";
        case TokenType::UNTIL: return "UNTIL";
        case TokenType::TYPE: return "TYPE";
        case TokenType::ENDTYPE: return "ENDTYPE";
        case TokenType::DOT: return "DOT";
        default: return "UNKNOWN";
    }
}


// Convert string to uppercase
std::string toUpper(const std::string& str) {
    std::string upperStr = str;
    for (char &c : upperStr) {
        c = std::toupper(static_cast<unsigned char>(c));
    }
    return upperStr;
}


// Tokenizer
std::vector<Token> tokenize(const std::string& input) {
    std::vector<Token> tokens;
    size_t i = 0;
    int line = 1, col = 1;


    std::map<std::string, TokenType> keywords {
        {"OUTPUT", TokenType::OUTPUT},
        {"INPUT", TokenType::INPUT},
        {"IF", TokenType::IF},
        {"ELSE", TokenType::ELSE},
        {"ENDIF", TokenType::ENDIF},
        {"DECLARE", TokenType::DECLARE},
        {"BOOLEAN", TokenType::BOOLEAN},
        {"INTEGER", TokenType::INTEGER},
        {"STRING", TokenType::STRING_TYPE},    // <-- New
        {"REAL", TokenType::REAL},             // <-- New
        {"ARRAY1D", TokenType::ARRAY1D},       // <-- New
        {"ARRAY2D", TokenType::ARRAY2D},       // <-- New
        {"THEN", TokenType::THEN},
        {"WAIT", TokenType::WAIT},
        {"FOR", TokenType::FOR},               // <-- New
        {"TO", TokenType::TO},                 // <-- New
        {"ENDFOR", TokenType::ENDFOR},         // <-- New
        {"TRUE", TokenType::TRUE},
        {"FALSE", TokenType::FALSE},
        {"YES", TokenType::YES},
        {"NO", TokenType::NO},
        {"WHILE", TokenType::WHILE},
        {"ENDWHILE", TokenType::ENDWHILE},
        {"CASE", TokenType::CASE},
        {"OF", TokenType::OF},
        {"OTHERWISE", TokenType::OTHERWISE},
        {"ENDCASE", TokenType::ENDCASE},
        {"REPEAT", TokenType::REPEAT},
        {"UNTIL", TokenType::UNTIL},
        {"TYPE", TokenType::TYPE},
        {"ENDTYPE", TokenType::ENDTYPE},
    };


    auto push = [&](TokenType t, const std::string& lex, int l, int c){
        tokens.push_back(Token{t, lex, l, c});
    };


    while (i < input.length()) {
        char c = input[i];
        if (c == '\r') { i++; continue; }
        if (c == '\n') { i++; line++; col = 1; continue; }
        if (std::isspace(static_cast<unsigned char>(c))) { i++; col++; continue; }
        if (c == '"') {
            int l = line, c0 = col;
            i++; col++;
            size_t start = i;
            std::string accum;
            bool closed = false;
            while (i < input.length()) {
                char d = input[i];
                if (d == '"') { closed = true; i++; col++; break; }
                if (d == '\n') { line++; col = 1; accum.push_back('\n'); i++; }
                else { accum.push_back(d); i++; col++; }
            }
            if (!closed) {
                throw std::runtime_error("Unclosed string literal at line " + std::to_string(l) + ", col " + std::to_string(c0));
            }
            push(TokenType::STRING, accum, l, c0);
            continue;
        }
        // Enhanced number parsing for real numbers
        if (std::isdigit(static_cast<unsigned char>(c))) {
            int l = line, c0 = col;
            size_t start = i;
            bool isReal = false;
            
            // Parse integer part
            while (i < input.length() && std::isdigit(static_cast<unsigned char>(input[i]))) {
                i++; col++;
            }
            
            // Check for decimal point
            if (i < input.length() && input[i] == '.') {
                isReal = true;
                i++; col++; // consume the '.'
                
                // Parse fractional part
                while (i < input.length() && std::isdigit(static_cast<unsigned char>(input[i]))) {
                    i++; col++;
                }
            }
            
            std::string numberStr = input.substr(start, i - start);
            push(isReal ? TokenType::REAL_NUMBER : TokenType::NUMBER, numberStr, l, c0);
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c))) {
            int l = line, c0 = col;
            size_t start = i;
            while (i < input.length() && std::isalnum(static_cast<unsigned char>(input[i]))) {
                i++; col++;
            }
            std::string lexeme = input.substr(start, i - start);
            std::string upperLexeme = toUpper(lexeme);
            if (keywords.count(upperLexeme)) {
                push(keywords[upperLexeme], lexeme, l, c0);
            } else {
                push(TokenType::IDENTIFIER, lexeme, l, c0);
            }
            continue;
        }
        if (c == ':' && i + 1 < input.length() && input[i + 1] == '=') {
            push(TokenType::ASSIGN, ":=", line, col);
            i += 2; col += 2; continue;
        }
        if (c == '>' && i + 1 < input.length() && input[i + 1] == '=') {
            push(TokenType::MOREOREQUAL, ">=", line, col);
            i += 2; col += 2; continue;
        }
        if (c == '<' && i + 1 < input.length() && input[i + 1] == '=') {
            push(TokenType::LESSOREQUAL, "<=", line, col);
            i += 2; col += 2; continue;
        }
        if (c == '!' && i + 1 < input.length() && input[i + 1] == '=') {
            push(TokenType::NOTEQUAL, "!=", line, col);
            i += 2; col += 2; continue;
        }
        switch (c) {
            case '+': push(TokenType::PLUS, "+", line, col); break;
            case '-': push(TokenType::MINUS, "-", line, col); break;
            case '*': push(TokenType::MULTIPLY, "*", line, col); break;
            case '/': push(TokenType::DIVIDE, "/", line, col); break;
            case '(': push(TokenType::LEFT_PAREN, "(", line, col); break;
            case ')': push(TokenType::RIGHT_PAREN, ")", line, col); break;
            case '[': push(TokenType::LEFT_BRACKET, "[", line, col); break;   // <-- New
            case ']': push(TokenType::RIGHT_BRACKET, "]", line, col); break; // <-- New
            case ',': push(TokenType::COMMA, ",", line, col); break;         // <-- New
            case '=': push(TokenType::EQUAL, "=", line, col); break;
            case '>': push(TokenType::MORE, ">", line, col); break;
            case '<': push(TokenType::LESS, "<", line, col); break;
            case ':': push(TokenType::COLON, ":",line, col); break;
            default:
                throw std::runtime_error("Unrecognized character '" + std::string(1, c) +
                                         "' at line " + std::to_string(line) +
                                         ", col " + std::to_string(col));
        }
        i++; col++;
    }
    tokens.push_back(Token{TokenType::ENDOFFILE, "", line, col});
    return tokens;
}


// --- Parser and AST Code ---


struct Node { virtual ~Node() = default; };


// Enhanced declaration node to support array dimensions
struct DeclarationNode : Node {
    Token identifier;
    Token dataType;
    std::unique_ptr<Node> initialValue;
    // For arrays: dimensions (1D has size1, 2D has size1 and size2)
    std::unique_ptr<Node> arraySize1;  // <-- New: first dimension
    std::unique_ptr<Node> arraySize2;  // <-- New: second dimension (for 2D arrays)
};


struct LiteralNode : Node { Token value; };


struct IdentifierNode : Node { Token identifier; };


// Enhanced to support array access
struct ArrayAccessNode : Node {  // <-- New: For array[index] or array[i][j]
    Token identifier;
    std::unique_ptr<Node> index1;     // First index
    std::unique_ptr<Node> index2;     // Second index (for 2D arrays, null for 1D)
};


struct OutputNode : Node { std::unique_ptr<Node> expression; };


struct InputNode : Node {
    Token identifier;
    std::unique_ptr<Node> index1;     // <-- New: For array input
    std::unique_ptr<Node> index2;     // <-- New: For 2D array input
};


struct AssignmentNode : Node {
    Token identifier;
    std::unique_ptr<Node> expression;
    std::unique_ptr<Node> index1;     // <-- New: For array assignment
    std::unique_ptr<Node> index2;     // <-- New: For 2D array assignment
};


struct UnaryOpNode : Node {
    Token op; // PLUS or MINUS
    std::unique_ptr<Node> operand;
};


struct BinaryOpNode : Node {
    std::unique_ptr<Node> left;
    Token op;
    std::unique_ptr<Node> right;
};


struct IfStatementNode : Node {
    std::unique_ptr<Node> condition;
    std::vector<std::unique_ptr<Node>> thenBody;
    std::vector<std::unique_ptr<Node>> elseBody;
};


struct WaitNode : Node {
    std::unique_ptr<Node> seconds;
};


// New: FOR loop node
struct ForLoopNode : Node {
    Token loopVariable;               // The loop counter variable (e.g., Count)
    std::unique_ptr<Node> startValue; // Start value (e.g., 1)
    std::unique_ptr<Node> endValue;   // End value (e.g., 5 or variable Num)
    std::vector<std::unique_ptr<Node>> body; // Statements inside the loop
};

//WHILE LOOP NODE
struct WhileLoopNode : Node {
    std::unique_ptr<Node> condition;
    std::vector<std::unique_ptr<Node>> body;
};

struct CaseItemNode : Node {
    std::unique_ptr<Node> value;
    std::vector<std::unique_ptr<Node>> body;
};

struct CaseOfNode : Node {
    std::unique_ptr<Node> expression;
    std::vector<std::unique_ptr<CaseItemNode>> cases;
    std::vector<std::unique_ptr<Node>> otherwiseBody;
};

struct RepeatLoopNode : Node {
    std::vector<std::unique_ptr<Node>> body;      // Statements inside the loop
    std::unique_ptr<Node> condition;              // UNTIL condition
};

class Parser {
public:
    Parser(const std::vector<Token>& tokens) : tokens(tokens), currentTokenIndex(0) {}


    std::vector<std::unique_ptr<Node>> parse() {
        std::vector<std::unique_ptr<Node>> program;
        while (peek().type != TokenType::ENDOFFILE) {
            program.push_back(parseStatement());
        }
        return program;
    }


private:
    const std::vector<Token>& tokens;
    size_t currentTokenIndex;


    const Token& peek(size_t lookahead = 0) {
        size_t idx = currentTokenIndex + lookahead;
        if (idx >= tokens.size()) {
            return tokens.back();
        }
        return tokens[idx];
    }


    const Token& consume(TokenType expectedType) {
        const Token& token = peek();
        if (token.type != expectedType) {
            throw std::runtime_error(
                "Unexpected token at line " + std::to_string(token.line) + ", col " + std::to_string(token.col) +
                ". Expected " + tokenTypeToString(expectedType) +
                " but got " + tokenTypeToString(token.type) +
                " with lexeme '" + token.lexeme + "'"
            );
        }
        currentTokenIndex++;
        return token;
    }


    bool match(TokenType t) {
        if (peek().type == t) { currentTokenIndex++; return true; }
        return false;
    }


    std::unique_ptr<Node> parseStatement() {
        switch (peek().type) {
            case TokenType::DECLARE: return parseDeclarationStatement();
            case TokenType::OUTPUT:  return parseOutputStatement();
            case TokenType::INPUT:   return parseInputStatement();
            case TokenType::IF:      return parseIfStatement();
            case TokenType::WAIT:    return parseWaitStatement();
            case TokenType::FOR:     return parseForStatement(); // <-- New
            case TokenType::WHILE:   return parseWhileStatement();
            case TokenType::CASE:    return parseCaseOfStatement();
            case TokenType::REPEAT:  return parseRepeatStatement();
            case TokenType::IDENTIFIER:
                if (peek(1).type == TokenType::ASSIGN || 
                    peek(1).type == TokenType::LEFT_BRACKET) {  // <-- Enhanced for arrays
                    return parseAssignmentStatement();
                }
            default: {
                const Token& t = peek();
                throw std::runtime_error(
                    "Unexpected statement start with token: " + tokenTypeToString(t.type) +
                    " at line " + std::to_string(t.line) + ", col " + std::to_string(t.col));
            }
        }
    }


    std::unique_ptr<Node> parseDeclarationStatement() {
        consume(TokenType::DECLARE);
        
        // Check for valid data types
        if (peek().type != TokenType::INTEGER && 
            peek().type != TokenType::BOOLEAN && 
            peek().type != TokenType::STRING_TYPE &&  // <-- New
            peek().type != TokenType::REAL &&         // <-- New
            peek().type != TokenType::ARRAY1D &&      // <-- New
            peek().type != TokenType::ARRAY2D) {      // <-- New
            const Token& t = peek();
            throw std::runtime_error("Invalid type in declaration at line " +
                                     std::to_string(t.line) + ", col " + std::to_string(t.col) +
                                     ". Expected INTEGER, BOOLEAN, STRING, REAL, ARRAY1D, or ARRAY2D but got " + 
                                     tokenTypeToString(t.type));
        }
        
        auto declarationNode = std::make_unique<DeclarationNode>();
        declarationNode->dataType = consume(peek().type);
        
        // Handle array dimensions
        if (declarationNode->dataType.type == TokenType::ARRAY1D) {
            consume(TokenType::LEFT_BRACKET);
            declarationNode->arraySize1 = parseExpression();
            consume(TokenType::RIGHT_BRACKET);
        } else if (declarationNode->dataType.type == TokenType::ARRAY2D) {
            consume(TokenType::LEFT_BRACKET);
            declarationNode->arraySize1 = parseExpression();
            consume(TokenType::COMMA);
            declarationNode->arraySize2 = parseExpression();
            consume(TokenType::RIGHT_BRACKET);
        }
        
        declarationNode->identifier = consume(TokenType::IDENTIFIER);
        consume(TokenType::ASSIGN);
        declarationNode->initialValue = parseExpression();
        return declarationNode;
    }


    std::unique_ptr<Node> parseAssignmentStatement() {
        auto assignmentNode = std::make_unique<AssignmentNode>();
        assignmentNode->identifier = consume(TokenType::IDENTIFIER);
        
        // Handle array indexing
        if (peek().type == TokenType::LEFT_BRACKET) {
            consume(TokenType::LEFT_BRACKET);
            assignmentNode->index1 = parseExpression();
            
            if (peek().type == TokenType::COMMA) {
                consume(TokenType::COMMA);
                assignmentNode->index2 = parseExpression();
            }
            
            consume(TokenType::RIGHT_BRACKET);
        }
        
        consume(TokenType::ASSIGN);
        assignmentNode->expression = parseExpression();
        return assignmentNode;
    }


    std::unique_ptr<Node> parseOutputStatement() {
        consume(TokenType::OUTPUT);
        auto outputNode = std::make_unique<OutputNode>();
        outputNode->expression = parseExpression();
        return outputNode;
    }


    std::unique_ptr<Node> parseInputStatement() {
        consume(TokenType::INPUT);
        auto inputNode = std::make_unique<InputNode>();
        inputNode->identifier = consume(TokenType::IDENTIFIER);
        
        // Handle array indexing for input
        if (peek().type == TokenType::LEFT_BRACKET) {
            consume(TokenType::LEFT_BRACKET);
            inputNode->index1 = parseExpression();
            
            if (peek().type == TokenType::COMMA) {
                consume(TokenType::COMMA);
                inputNode->index2 = parseExpression();
            }
            
            consume(TokenType::RIGHT_BRACKET);
        }
        
        return inputNode;
    }


    std::unique_ptr<Node> parseWaitStatement() {
        consume(TokenType::WAIT);
        auto waitNode = std::make_unique<WaitNode>();
        waitNode->seconds = parseExpression();
        return waitNode;
    }


    // New: Parse FOR statement
    std::unique_ptr<Node> parseForStatement() {
        consume(TokenType::FOR);
        
        auto forNode = std::make_unique<ForLoopNode>();
        
        // Parse loop variable (e.g., Count)
        forNode->loopVariable = consume(TokenType::IDENTIFIER);
        
        // Parse assignment (:=)
        consume(TokenType::ASSIGN);
        
        // Parse start value
        forNode->startValue = parseExpression();
        
        // Parse TO keyword
        consume(TokenType::TO);
        
        // Parse end value (can be literal or variable)
        forNode->endValue = parseExpression();
        
        // Parse loop body until ENDFOR
        while (peek().type != TokenType::ENDFOR && peek().type != TokenType::ENDOFFILE) {
            forNode->body.push_back(parseStatement());
        }
        
        consume(TokenType::ENDFOR);
        
        return forNode;
    }


    // Expression parsing remains the same structure
    std::unique_ptr<Node> parseExpression() { return parseEquality(); }


    std::unique_ptr<Node> parseEquality() {
        auto left = parseComparison();
        while (peek().type == TokenType::EQUAL || peek().type == TokenType::NOTEQUAL) {
            Token op = consume(peek().type);
            auto right = parseComparison();
            auto node = std::make_unique<BinaryOpNode>();
            node->left = std::move(left);
            node->op = op;
            node->right = std::move(right);
            left = std::move(node);
        }
        return left;
    }


    std::unique_ptr<Node> parseComparison() {
        auto left = parseAdditive();
        while (peek().type == TokenType::MORE ||
               peek().type == TokenType::LESS ||
               peek().type == TokenType::MOREOREQUAL ||
               peek().type == TokenType::LESSOREQUAL) {
            Token op = consume(peek().type);
            auto right = parseAdditive();
            auto node = std::make_unique<BinaryOpNode>();
            node->left = std::move(left);
            node->op = op;
            node->right = std::move(right);
            left = std::move(node);
        }
        return left;
    }


    std::unique_ptr<Node> parseAdditive() {
        auto left = parseMultiplicative();
        while (peek().type == TokenType::PLUS || peek().type == TokenType::MINUS) {
            Token op = consume(peek().type);
            auto right = parseMultiplicative();
            auto node = std::make_unique<BinaryOpNode>();
            node->left = std::move(left);
            node->op = op;
            node->right = std::move(right);
            left = std::move(node);
        }
        return left;
    }


    std::unique_ptr<Node> parseMultiplicative() {
        auto left = parseUnary();
        while (peek().type == TokenType::MULTIPLY || peek().type == TokenType::DIVIDE) {
            Token op = consume(peek().type);
            auto right = parseUnary();
            auto node = std::make_unique<BinaryOpNode>();
            node->left = std::move(left);
            node->op = op;
            node->right = std::move(right);
            left = std::move(node);
        }
        return left;
    }


    std::unique_ptr<Node> parseUnary() {
        if (peek().type == TokenType::PLUS || peek().type == TokenType::MINUS) {
            Token op = consume(peek().type);
            auto operand = parseUnary();
            auto node = std::make_unique<UnaryOpNode>();
            node->op = op;
            node->operand = std::move(operand);
            return node;
        }
        return parsePrimary();
    }


    std::unique_ptr<Node> parsePrimary() {
        const Token& token = peek();
        switch (token.type) {
            case TokenType::NUMBER:
            case TokenType::REAL_NUMBER:    // <-- New
            case TokenType::STRING:
            case TokenType::TRUE:
            case TokenType::FALSE:
            case TokenType::YES:
            case TokenType::NO: {
                consume(token.type);
                auto node = std::make_unique<LiteralNode>();
                node->value = token;
                return node;
            }
            case TokenType::IDENTIFIER: {
                Token id = consume(TokenType::IDENTIFIER);
                
                // Check for array access
                if (peek().type == TokenType::LEFT_BRACKET) {
                    auto arrayNode = std::make_unique<ArrayAccessNode>();
                    arrayNode->identifier = id;
                    
                    consume(TokenType::LEFT_BRACKET);
                    arrayNode->index1 = parseExpression();
                    
                    if (peek().type == TokenType::COMMA) {
                        consume(TokenType::COMMA);
                        arrayNode->index2 = parseExpression();
                    }
                    
                    consume(TokenType::RIGHT_BRACKET);
                    return arrayNode;
                } else {
                    auto node = std::make_unique<IdentifierNode>();
                    node->identifier = id;
                    return node;
                }
            }
            case TokenType::LEFT_PAREN: {
                consume(TokenType::LEFT_PAREN);
                auto expr = parseExpression();
                consume(TokenType::RIGHT_PAREN);
                return expr;
            }
            default:
                throw std::runtime_error("Expected an expression at line " +
                                         std::to_string(token.line) + ", col " +
                                         std::to_string(token.col) + " but got " +
                                         tokenTypeToString(token.type));
        }
    }


    std::unique_ptr<Node> parseIfStatement() {
        consume(TokenType::IF);
        auto ifNode = std::make_unique<IfStatementNode>();
        ifNode->condition = parseExpression();
        consume(TokenType::THEN);
        while (peek().type != TokenType::ELSE && peek().type != TokenType::ENDIF) {
            ifNode->thenBody.push_back(parseStatement());
        }
        if (peek().type == TokenType::ELSE) {
            consume(TokenType::ELSE);
            while (peek().type != TokenType::ENDIF) {
                ifNode->elseBody.push_back(parseStatement());
            }
        }
        consume(TokenType::ENDIF);
        return ifNode;
    }

    std::unique_ptr<Node> parseWhileStatement() {
        consume(TokenType::WHILE);
        
        auto whileNode = std::make_unique<WhileLoopNode>();
        whileNode->condition = parseExpression();
        
        // Parse loop body until ENDWHILE
        while (peek().type != TokenType::ENDWHILE && peek().type != TokenType::ENDOFFILE) {
            whileNode->body.push_back(parseStatement());
        }
        
        consume(TokenType::ENDWHILE);
        return whileNode;
    }

    std::unique_ptr<Node> parseCaseOfStatement() {
        consume(TokenType::CASE);
        consume(TokenType::OF);

        auto caseNode = std::make_unique<CaseOfNode>();
        caseNode -> expression = parseExpression();

        while(peek(). type != TokenType::OTHERWISE && peek().type != TokenType::ENDCASE && peek().type != TokenType::ENDOFFILE) {
            auto caseItem = std::make_unique<CaseItemNode>();

            caseItem -> value = parseExpression();
            consume(TokenType::COLON);

            while (peek().type != TokenType::OTHERWISE && peek().type != TokenType::ENDCASE && peek().type != TokenType::ENDOFFILE) {
                if (peek().type == TokenType::IDENTIFIER && peek(1).type == TokenType::COLON) {
                    break;
                }
                caseItem -> body.push_back(parseStatement());
            }
            caseNode -> cases.push_back(std::move(caseItem));
        };

        if (peek().type == TokenType::OTHERWISE) {
            consume(TokenType::OTHERWISE);
            consume(TokenType::COLON);

            while (peek().type != TokenType::ENDCASE && peek().type != TokenType::ENDOFFILE) {
                caseNode -> otherwiseBody.push_back(parseStatement());
            }
        };

        consume(TokenType::ENDCASE);
        return caseNode;
    }

    std::unique_ptr<Node> parseRepeatStatement() {
        consume(TokenType::REPEAT);
        
        auto repeatNode = std::make_unique<RepeatLoopNode>();
        
        // Parse loop body until UNTIL
        while (peek().type != TokenType::UNTIL && peek().type != TokenType::ENDOFFILE) {
            repeatNode->body.push_back(parseStatement());
        }
        
        consume(TokenType::UNTIL);
        
        // Parse condition
        repeatNode->condition = parseExpression();
        
        return repeatNode;
    }
};


// Enhanced Symbol Table to support new data types
class SymbolTable {
public:
    struct VariableInfo {
        TokenType type;
        bool isArray1D = false;
        bool isArray2D = false;
        int size1 = 0;  // First dimension size
        int size2 = 0;  // Second dimension size
    };

    void addVariable(const std::string& name, TokenType type, bool isArray1D = false, bool isArray2D = false, int size1 = 0, int size2 = 0) {
        if (scopes.empty()) {
            throw std::runtime_error("Cannot add variable, no scope active.");
        }
        auto& currentScope = scopes.back();
        if (currentScope.count(name)) {
            throw std::runtime_error("Variable '" + name + "' already declared in this scope.");
        }
        
        VariableInfo info;
        info.type = type;
        info.isArray1D = isArray1D;
        info.isArray2D = isArray2D;
        info.size1 = size1;
        info.size2 = size2;
        
        currentScope[name] = info;
    }


    VariableInfo getVariableInfo(const std::string& name) {
        // Search from the innermost scope outwards
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            if (it->count(name)) {
                return it->at(name);
            }
        }
        throw std::runtime_error("Variable '" + name + "' not declared.");
    }

    TokenType getVariableType(const std::string& name) {
        return getVariableInfo(name).type;
    }


    // Manage scopes
    void pushScope() {
        scopes.emplace_back();
    }


    void popScope() {
        if (!scopes.empty()) {
            scopes.pop_back();
        }
    }


private:
    std::vector<std::map<std::string, VariableInfo>> scopes;
};


// Enhanced semantic analyzer
static inline TokenType normalize(TokenType t) {
    if (t == TokenType::NUMBER) return TokenType::INTEGER;
    if (t == TokenType::REAL_NUMBER) return TokenType::REAL;  // <-- New
    if (t == TokenType::TRUE || t == TokenType::FALSE || t == TokenType::YES || t == TokenType::NO) return TokenType::BOOLEAN;
    return t;
}


static inline bool isNumeric(TokenType t) {
    return t == TokenType::INTEGER || t == TokenType::REAL;  // <-- Enhanced
}


static inline bool isEqualityOp(TokenType t) {
    return t == TokenType::EQUAL || t == TokenType::NOTEQUAL;
}


static inline bool isComparisonOp(TokenType t) {
    return t == TokenType::MORE || t == TokenType::LESS || t == TokenType::MOREOREQUAL || t == TokenType::LESSOREQUAL;
}


static inline bool isArithmeticOp(TokenType t) {
    return t == TokenType::PLUS || t == TokenType::MINUS || t == TokenType::MULTIPLY || t == TokenType::DIVIDE;
}


class SemanticAnalyzer {
public:
    void analyze(const std::vector<std::unique_ptr<Node>>& ast) {
        // Start with the global scope
        symbolTable.pushScope();
        for (const auto& node : ast) {
            visitStatement(node.get());
        }
        // Pop the global scope
        symbolTable.popScope();
    }


private:
    SymbolTable symbolTable;


    void visitStatement(Node* node) {
        if (!node) return;
        if (auto* decl = dynamic_cast<DeclarationNode*>(node))     visitDeclaration(decl);
        else if (auto* output = dynamic_cast<OutputNode*>(node))   visitOutput(output);
        else if (auto* ifStmt = dynamic_cast<IfStatementNode*>(node)) visitIfStatement(ifStmt);
        else if (auto* input = dynamic_cast<InputNode*>(node))    visitInput(input);
        else if (auto* assign = dynamic_cast<AssignmentNode*>(node)) visitAssignment(assign);
        else if (auto* wait = dynamic_cast<WaitNode*>(node))      visitWait(wait);
        else if (auto* forLoop = dynamic_cast<ForLoopNode*>(node)) visitForLoop(forLoop);  // <-- New
        else if (auto* whileLoop = dynamic_cast<WhileLoopNode*>(node)) visitWhileLoop(whileLoop);
        else if (auto* caseStmt = dynamic_cast<CaseOfNode*>(node)) visitCaseOf(caseStmt);
        else if (auto* repeatLoop = dynamic_cast<RepeatLoopNode*>(node)) visitRepeatLoop(repeatLoop);
    }


    void visitDeclaration(DeclarationNode* node) {
        TokenType initType = normalize(visitExpression(node->initialValue.get()));
        TokenType declared = node->dataType.type;

        // Handle array declarations
        if (declared == TokenType::ARRAY1D) {
            // For 1D arrays, we need to specify the element type and default to 0
            // For now, assume integer arrays (could be extended)
            TokenType elementType = TokenType::INTEGER; // Default element type
            
            // Get array size
            TokenType sizeType = normalize(visitExpression(node->arraySize1.get()));
            if (!isNumeric(sizeType)) {
                throw std::runtime_error("Array size must be numeric in declaration of '" + node->identifier.lexeme + "'");
            }
            
            symbolTable.addVariable(node->identifier.lexeme, elementType, true, false, 0, 0); // Size will be determined at runtime
            
        } else if (declared == TokenType::ARRAY2D) {
            // For 2D arrays
            TokenType elementType = TokenType::INTEGER; // Default element type
            
            // Get array sizes
            TokenType size1Type = normalize(visitExpression(node->arraySize1.get()));
            TokenType size2Type = normalize(visitExpression(node->arraySize2.get()));
            
            if (!isNumeric(size1Type) || !isNumeric(size2Type)) {
                throw std::runtime_error("Array dimensions must be numeric in declaration of '" + node->identifier.lexeme + "'");
            }
            
            symbolTable.addVariable(node->identifier.lexeme, elementType, false, true, 0, 0); // Sizes determined at runtime
            
        } else {
            // Handle regular variable types
            if (declared == TokenType::INTEGER) {
                if (!isNumeric(initType)) {
                    throw std::runtime_error("Type mismatch in declaration of '" + node->identifier.lexeme +
                                             "'. Expected INTEGER but got " + tokenTypeToString(initType));
                }
            } else if (declared == TokenType::REAL) {  // <-- New
                if (!isNumeric(initType)) {
                    throw std::runtime_error("Type mismatch in declaration of '" + node->identifier.lexeme +
                                             "'. Expected REAL but got " + tokenTypeToString(initType));
                }
            } else if (declared == TokenType::BOOLEAN) {
                if (initType != TokenType::BOOLEAN) {
                    throw std::runtime_error("Type mismatch in declaration of '" + node->identifier.lexeme +
                                             "'. Expected BOOLEAN but got " + tokenTypeToString(initType));
                }
            } else if (declared == TokenType::STRING_TYPE) {  // <-- New
                if (initType != TokenType::STRING) {
                    throw std::runtime_error("Type mismatch in declaration of '" + node->identifier.lexeme +
                                             "'. Expected STRING but got " + tokenTypeToString(initType));
                }
            } else {
                throw std::runtime_error("Unsupported declared type: " + tokenTypeToString(declared));
            }
            
            symbolTable.addVariable(node->identifier.lexeme, declared);
        }
    }


    void visitOutput(OutputNode* node) {
        TokenType t = normalize(visitExpression(node->expression.get()));
        if (t != TokenType::INTEGER && t != TokenType::REAL && t != TokenType::BOOLEAN && t != TokenType::STRING) {
            throw std::runtime_error("OUTPUT expects STRING, INTEGER, REAL, or BOOLEAN expression.");
        }
    }


    void visitIfStatement(IfStatementNode* node) {
        TokenType conditionType = normalize(visitExpression(node->condition.get()));
        if (conditionType != TokenType::BOOLEAN) {
            throw std::runtime_error("IF statement condition must be a BOOLEAN expression.");
        }
        // Scope for the 'then' block
        symbolTable.pushScope();
        for (const auto& stmt : node->thenBody)  visitStatement(stmt.get());
        symbolTable.popScope();


        // Scope for the 'else' block
        symbolTable.pushScope();
        for (const auto& stmt : node->elseBody)  visitStatement(stmt.get());
        symbolTable.popScope();
    }


    // New: FOR loop semantic analysis
    void visitForLoop(ForLoopNode* node) {
        // Check that start and end values are numeric
        TokenType startType = normalize(visitExpression(node->startValue.get()));
        TokenType endType = normalize(visitExpression(node->endValue.get()));
        
        if (!isNumeric(startType)) {
            throw std::runtime_error("FOR loop start value must be numeric.");
        }
        if (!isNumeric(endType)) {
            throw std::runtime_error("FOR loop end value must be numeric.");
        }
        
        // Create a new scope for the loop
        symbolTable.pushScope();
        
        // Add the loop variable as an INTEGER to the loop scope
        symbolTable.addVariable(node->loopVariable.lexeme, TokenType::INTEGER);
        
        // Analyze the loop body
        for (const auto& stmt : node->body) {
            visitStatement(stmt.get());
        }
        
        // Pop the loop scope
        symbolTable.popScope();
    }


    void visitInput(InputNode* node) {
        try {
            SymbolTable::VariableInfo info = symbolTable.getVariableInfo(node->identifier.lexeme);
            
            // Check array access for input
            if (node->index1) {
                if (!info.isArray1D && !info.isArray2D) {
                    throw std::runtime_error("Variable '" + node->identifier.lexeme + "' is not an array but used with array indexing in INPUT.");
                }
                
                TokenType indexType = normalize(visitExpression(node->index1.get()));
                if (!isNumeric(indexType)) {
                    throw std::runtime_error("Array index must be numeric in INPUT statement.");
                }
                
                if (node->index2) {
                    if (!info.isArray2D) {
                        throw std::runtime_error("Variable '" + node->identifier.lexeme + "' is not a 2D array but used with 2D indexing in INPUT.");
                    }
                    
                    TokenType index2Type = normalize(visitExpression(node->index2.get()));
                    if (!isNumeric(index2Type)) {
                        throw std::runtime_error("Second array index must be numeric in INPUT statement.");
                    }
                }
            }
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("Variable '" + node->identifier.lexeme + "' used in INPUT statement has not been declared.");
        }
    }

    void visitRepeatLoop(RepeatLoopNode* node) {
        // Create new scope for loop body
        symbolTable.pushScope();
        
        // Analyze loop body first (since it executes before condition check)
        for (const auto& stmt : node->body) {
            visitStatement(stmt.get());
        }
        
        // Check that condition evaluates to boolean
        TokenType conditionType = normalize(visitExpression(node->condition.get()));
        if (conditionType != TokenType::BOOLEAN) {
            throw std::runtime_error("REPEAT loop UNTIL condition must be a BOOLEAN expression.");
        }
        
        symbolTable.popScope();
    }


    void visitAssignment(AssignmentNode* node) {
        SymbolTable::VariableInfo varInfo;
        try {
            varInfo = symbolTable.getVariableInfo(node->identifier.lexeme);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("Variable '" + node->identifier.lexeme + "' used in assignment has not been declared.");
        }
        
        TokenType valueType = normalize(visitExpression(node->expression.get()));
        TokenType varType = normalize(varInfo.type);

        // Check array access for assignment
        if (node->index1) {
            if (!varInfo.isArray1D && !varInfo.isArray2D) {
                throw std::runtime_error("Variable '" + node->identifier.lexeme + "' is not an array but used with array indexing in assignment.");
            }
            
            TokenType indexType = normalize(visitExpression(node->index1.get()));
            if (!isNumeric(indexType)) {
                throw std::runtime_error("Array index must be numeric in assignment.");
            }
            
            if (node->index2) {
                if (!varInfo.isArray2D) {
                    throw std::runtime_error("Variable '" + node->identifier.lexeme + "' is not a 2D array but used with 2D indexing.");
                }
                
                TokenType index2Type = normalize(visitExpression(node->index2.get()));
                if (!isNumeric(index2Type)) {
                    throw std::runtime_error("Second array index must be numeric in assignment.");
                }
            }
        }

        // Type compatibility checking
        if (varType == TokenType::INTEGER) {
            if (!isNumeric(valueType)) {
                throw std::runtime_error("Type mismatch in assignment to '" + node->identifier.lexeme +
                                         "'. Expected INTEGER but got " + tokenTypeToString(valueType));
            }
        } else if (varType == TokenType::REAL) {  // <-- New
            if (!isNumeric(valueType)) {
                throw std::runtime_error("Type mismatch in assignment to '" + node->identifier.lexeme +
                                         "'. Expected REAL but got " + tokenTypeToString(valueType));
            }
        } else if (varType == TokenType::BOOLEAN) {
            if (valueType != TokenType::BOOLEAN) {
                throw std::runtime_error("Type mismatch in assignment to '" + node->identifier.lexeme +
                                         "'. Expected BOOLEAN but got " + tokenTypeToString(valueType));
            }
        } else if (varType == TokenType::STRING_TYPE) {  // <-- New
            if (valueType != TokenType::STRING) {
                throw std::runtime_error("Type mismatch in assignment to '" + node->identifier.lexeme +
                                         "'. Expected STRING but got " + tokenTypeToString(valueType));
            }
        }
    }


    void visitWait(WaitNode* node) {
        TokenType secondsType = normalize(visitExpression(node->seconds.get()));
        if (!isNumeric(secondsType)) {
            throw std::runtime_error("WAIT statement requires a numeric expression for seconds.");
        }
    }

    void visitWhileLoop(WhileLoopNode* node) {
        // Check that condition evaluates to boolean
        TokenType conditionType = normalize(visitExpression(node->condition.get()));
        if (conditionType != TokenType::BOOLEAN) {
            throw std::runtime_error("WHILE loop condition must be a BOOLEAN expression.");
        }
        
        // Create new scope for loop body
        symbolTable.pushScope();
        for (const auto& stmt : node->body) {
            visitStatement(stmt.get());
        }
        symbolTable.popScope();
    }

    void visitCaseOf(CaseOfNode* node) {
        // Get the type of the expression being switched on
        TokenType switchExprType = normalize(visitExpression(node->expression.get()));
        
        // Check each case value matches the switch expression type
        for (const auto& caseItem : node->cases) {
            TokenType caseValueType = normalize(visitExpression(caseItem->value.get()));
            
            // Allow some flexibility - identifiers, strings, numbers should all work
            if (switchExprType == TokenType::IDENTIFIER) {
                // If switching on a variable, case values should be compatible
                if (caseValueType != TokenType::IDENTIFIER && caseValueType != TokenType::STRING && !isNumeric(caseValueType)) {
                    throw std::runtime_error("CASE value type incompatible with CASE OF expression type.");
                }
            } else if (switchExprType != caseValueType) {
                throw std::runtime_error("CASE value type doesn't match CASE OF expression type.");
            }
            
            // Analyze case body
            symbolTable.pushScope();
            for (const auto& stmt : caseItem->body) {
                visitStatement(stmt.get());
            }
            symbolTable.popScope();
        }
        
        // Analyze OTHERWISE body if present
        if (!node->otherwiseBody.empty()) {
            symbolTable.pushScope();
            for (const auto& stmt : node->otherwiseBody) {
                visitStatement(stmt.get());
            }
            symbolTable.popScope();
        }
    }

    TokenType visitExpression(Node* node) {
        if (!node) throw std::runtime_error("Empty expression node found.");
        
        if (auto* lit = dynamic_cast<LiteralNode*>(node)) {
            return lit->value.type;
        }
        
        if (auto* id = dynamic_cast<IdentifierNode*>(node)) {
            return symbolTable.getVariableType(id->identifier.lexeme);
        }
        
        if (auto* arrayAccess = dynamic_cast<ArrayAccessNode*>(node)) {  // <-- New
            SymbolTable::VariableInfo info = symbolTable.getVariableInfo(arrayAccess->identifier.lexeme);
            
            if (!info.isArray1D && !info.isArray2D) {
                throw std::runtime_error("Variable '" + arrayAccess->identifier.lexeme + "' is not an array.");
            }
            
            TokenType indexType = normalize(visitExpression(arrayAccess->index1.get()));
            if (!isNumeric(indexType)) {
                throw std::runtime_error("Array index must be numeric.");
            }
            
            if (arrayAccess->index2) {
                if (!info.isArray2D) {
                    throw std::runtime_error("Variable '" + arrayAccess->identifier.lexeme + "' is not a 2D array.");
                }
                
                TokenType index2Type = normalize(visitExpression(arrayAccess->index2.get()));
                if (!isNumeric(index2Type)) {
                    throw std::runtime_error("Second array index must be numeric.");
                }
            }
            
            return info.type; // Return the element type
        }
        
        if (auto* un = dynamic_cast<UnaryOpNode*>(node)) {
            TokenType t = normalize(visitExpression(un->operand.get()));
            if (un->op.type == TokenType::PLUS || un->op.type == TokenType::MINUS) {
                if (!isNumeric(t)) {
                    throw std::runtime_error("Unary operator '" + un->op.lexeme + "' requires numeric operand.");
                }
                return t; // Return the same numeric type (INTEGER or REAL)
            }
            throw std::runtime_error("Unsupported unary operator: " + un->op.lexeme);
        }
        
        if (auto* bin = dynamic_cast<BinaryOpNode*>(node)) {
            TokenType lt = normalize(visitExpression(bin->left.get()));
            TokenType rt = normalize(visitExpression(bin->right.get()));


            if (isArithmeticOp(bin->op.type) || isComparisonOp(bin->op.type)) {
                if (!isNumeric(lt) || !isNumeric(rt)) {
                    throw std::runtime_error("Operator '" + bin->op.lexeme + "' requires numeric operands.");
                }
            }
           
            if (isArithmeticOp(bin->op.type)) {
                // Return REAL if either operand is REAL, otherwise INTEGER
                if (lt == TokenType::REAL || rt == TokenType::REAL) {
                    return TokenType::REAL;
                }
                return TokenType::INTEGER;
            }


            if (isComparisonOp(bin->op.type) || isEqualityOp(bin->op.type)) {
                // Allow comparison between INTEGER and REAL
                if (isNumeric(lt) && isNumeric(rt)) {
                    return TokenType::BOOLEAN;
                }
                if (lt != rt) {
                    throw std::runtime_error("Type mismatch in expression: left is " +
                                             tokenTypeToString(lt) + ", right is " + tokenTypeToString(rt));
                }
                return TokenType::BOOLEAN;
            }
        }
        throw std::runtime_error("Unsupported expression type during semantic analysis.");
    }
};


// Enhanced Code Generator
class CodeGenerator {
public:
    void generate(const std::vector<std::unique_ptr<Node>>& ast, std::ostream& output) {
        // C++ boilerplate: includes and main function.
        output << "#include <iostream>\n";
        output << "#include <string>\n";
        output << "#include <vector>\n";  // <-- New: For arrays
        output << "#include <stdexcept>\n";
        output << "#include <chrono>\n";
        output << "#include <thread>\n\n";
        output << "int main() {\n";
       
        for (const auto& node : ast) {
            visit(node.get(), output, 1);
        }
       
        output << "    return 0;\n";
        output << "}\n";
    }


private:
    void indent(std::ostream& output, int level) {
        for (int i = 0; i < level; ++i) {
            output << "    ";
        }
    }


    void visit(Node* node, std::ostream& output, int indentLevel) {
        if (!node) return;
        if (auto* decl = dynamic_cast<DeclarationNode*>(node)) {
            visitDeclaration(decl, output, indentLevel);
        } else if (auto* outputStmt = dynamic_cast<OutputNode*>(node)) {
            visitOutput(outputStmt, output, indentLevel);
        } else if (auto* inputStmt = dynamic_cast<InputNode*>(node)) {
            visitInput(inputStmt, output, indentLevel);
        } else if (auto* assign = dynamic_cast<AssignmentNode*>(node)) {
            visitAssignment(assign, output, indentLevel);
        } else if (auto* ifStmt = dynamic_cast<IfStatementNode*>(node)) {
            visitIfStatement(ifStmt, output, indentLevel);
        } else if (auto* waitStmt = dynamic_cast<WaitNode*>(node)) {
            visitWait(waitStmt, output, indentLevel);
        } else if (auto* forLoop = dynamic_cast<ForLoopNode*>(node)) {  // <-- New
            visitForLoop(forLoop, output, indentLevel);
        } else if (auto* whileLoop = dynamic_cast<WhileLoopNode*>(node)) {
            visitWhileLoop(whileLoop, output, indentLevel);
        } else if (auto* caseStmt = dynamic_cast<CaseOfNode*>(node)) {
            visitCaseOf(caseStmt, output, indentLevel);  // <-- Add this
        } else if (auto* repeatLoop = dynamic_cast<RepeatLoopNode*>(node)) {
            visitRepeatLoop(repeatLoop, output, indentLevel);
        }
    }


    void visitDeclaration(DeclarationNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        
        if (node->dataType.type == TokenType::INTEGER) {
            output << "int " << node->identifier.lexeme;
        } else if (node->dataType.type == TokenType::REAL) {  // <-- New
            output << "double " << node->identifier.lexeme;
        } else if (node->dataType.type == TokenType::BOOLEAN) {
            output << "bool " << node->identifier.lexeme;
        } else if (node->dataType.type == TokenType::STRING_TYPE) {  // <-- New
            output << "std::string " << node->identifier.lexeme;
        } else if (node->dataType.type == TokenType::ARRAY1D) {  // <-- New
            output << "std::vector<int> " << node->identifier.lexeme << "(";
            visitExpression(node->arraySize1.get(), output);
            output << ", ";
            visitExpression(node->initialValue.get(), output);
            output << ");\n";
            return; // Early return to avoid the assignment below
        } else if (node->dataType.type == TokenType::ARRAY2D) {  // <-- New
            output << "std::vector<std::vector<int>> " << node->identifier.lexeme << "(";
            visitExpression(node->arraySize1.get(), output);
            output << ", std::vector<int>(";
            visitExpression(node->arraySize2.get(), output);
            output << ", ";
            visitExpression(node->initialValue.get(), output);
            output << "));\n";
            return; // Early return to avoid the assignment below
        }
        
        output << " = ";
        visitExpression(node->initialValue.get(), output);
        output << ";\n";
    }


    void visitAssignment(AssignmentNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << node->identifier.lexeme;
        
        // Handle array indexing
        if (node->index1) {
            output << "[";
            visitExpression(node->index1.get(), output);
            output << "]";
            
            if (node->index2) {
                output << "[";
                visitExpression(node->index2.get(), output);
                output << "]";
            }
        }
        
        output << " = ";
        visitExpression(node->expression.get(), output);
        output << ";\n";
    }


    void visitOutput(OutputNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << "std::cout << ";
        visitExpression(node->expression.get(), output);
        output << " << std::endl;\n";
    }


    void visitInput(InputNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << "std::cin >> " << node->identifier.lexeme;
        
        // Handle array indexing for input
        if (node->index1) {
            output << "[";
            visitExpression(node->index1.get(), output);
            output << "]";
            
            if (node->index2) {
                output << "[";
                visitExpression(node->index2.get(), output);
                output << "]";
            }
        }
        
        output << ";\n";
    }
   
    void visitWait(WaitNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << "std::this_thread::sleep_for(std::chrono::seconds(";
        visitExpression(node->seconds.get(), output);
        output << "));\n";
    }


    // New: FOR loop code generation
    void visitForLoop(ForLoopNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << "for (int " << node->loopVariable.lexeme << " = ";
        visitExpression(node->startValue.get(), output);
        output << "; " << node->loopVariable.lexeme << " <= ";
        visitExpression(node->endValue.get(), output);
        output << "; ++" << node->loopVariable.lexeme << ") {\n";
        
        // Generate loop body
        for (const auto& stmt : node->body) {
            visit(stmt.get(), output, indentLevel + 1);
        }
        
        indent(output, indentLevel);
        output << "}\n";
    }
   
    void visitIfStatement(IfStatementNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << "if (";
        visitExpression(node->condition.get(), output);
        output << ") {\n";
        for (const auto& stmt : node->thenBody) {
            visit(stmt.get(), output, indentLevel + 1);
        }
        indent(output, indentLevel);
        output << "}\n";
        if (!node->elseBody.empty()) {
            indent(output, indentLevel);
            output << "else {\n";
            for (const auto& stmt : node->elseBody) {
                visit(stmt.get(), output, indentLevel + 1);
            }
            indent(output, indentLevel);
            output << "}\n";
        }
    }


    void visitExpression(Node* node, std::ostream& output) {
        if (!node) return;
        if (auto* lit = dynamic_cast<LiteralNode*>(node)) {
            visitLiteral(lit, output);
        } else if (auto* id = dynamic_cast<IdentifierNode*>(node)) {
            visitIdentifier(id, output);
        } else if (auto* arrayAccess = dynamic_cast<ArrayAccessNode*>(node)) {  // <-- New
            visitArrayAccess(arrayAccess, output);
        } else if (auto* un = dynamic_cast<UnaryOpNode*>(node)) {
            visitUnaryOp(un, output);
        } else if (auto* bin = dynamic_cast<BinaryOpNode*>(node)) {
            visitBinaryOp(bin, output);
        }
    }
   
    void visitLiteral(LiteralNode* node, std::ostream& output) {
        switch (node->value.type) {
            case TokenType::NUMBER:
                output << node->value.lexeme;
                break;
            case TokenType::REAL_NUMBER:  // <-- New
                output << node->value.lexeme;
                break;
            case TokenType::STRING:
                output << "\"" << node->value.lexeme << "\"";
                break;
            case TokenType::TRUE:
            case TokenType::YES:
                output << "true";
                break;
            case TokenType::FALSE:
            case TokenType::NO:
                output << "false";
                break;
            default:
                throw std::runtime_error("Unsupported literal type for code generation.");
        }
    }


    void visitIdentifier(IdentifierNode* node, std::ostream& output) {
        output << node->identifier.lexeme;
    }
    
    void visitArrayAccess(ArrayAccessNode* node, std::ostream& output) {  // <-- New
        output << node->identifier.lexeme << "[";
        visitExpression(node->index1.get(), output);
        output << "]";
        
        if (node->index2) {
            output << "[";
            visitExpression(node->index2.get(), output);
            output << "]";
        }
    }


    void visitUnaryOp(UnaryOpNode* node, std::ostream& output) {
        output << node->op.lexeme;
        visitExpression(node->operand.get(), output);
    }
   
    void visitBinaryOp(BinaryOpNode* node, std::ostream& output) {
        output << "(";
        visitExpression(node->left.get(), output);
        output << " " << mapOperator(node->op.type) << " ";
        visitExpression(node->right.get(), output);
        output << ")";
    }

    void visitWhileLoop(WhileLoopNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << "while (";
        visitExpression(node->condition.get(), output);
        output << ") {\n";
        
        // Generate loop body
        for (const auto& stmt : node->body) {
            visit(stmt.get(), output, indentLevel + 1);
        }
        
        indent(output, indentLevel);
        output << "}\n";
    }

    void visitCaseOf(CaseOfNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << "{\n";  // Create a block for the switch logic
        
        // Generate variable to hold the switch expression value
        indent(output, indentLevel + 1);
        output << "auto __switch_value = ";
        visitExpression(node->expression.get(), output);
        output << ";\n";
        
        // Generate if-else chain for cases
        bool firstCase = true;
        for (const auto& caseItem : node->cases) {
            indent(output, indentLevel + 1);
            if (firstCase) {
                output << "if (__switch_value == ";
                firstCase = false;
            } else {
                output << "else if (__switch_value == ";
            }
            
            visitExpression(caseItem->value.get(), output);
            output << ") {\n";
            
            // Generate case body
            for (const auto& stmt : caseItem->body) {
                visit(stmt.get(), output, indentLevel + 2);
            }
            
            indent(output, indentLevel + 1);
            output << "}\n";
        }
        
        // Generate OTHERWISE clause as final else
        if (!node->otherwiseBody.empty()) {
            indent(output, indentLevel + 1);
            output << "else {\n";
            
            for (const auto& stmt : node->otherwiseBody) {
                visit(stmt.get(), output, indentLevel + 2);
            }
            
            indent(output, indentLevel + 1);
            output << "}\n";
        }
        
        indent(output, indentLevel);
        output << "}\n";
    }

    void visitRepeatLoop(RepeatLoopNode* node, std::ostream& output, int indentLevel) {
        indent(output, indentLevel);
        output << "do {\n";
        
        // Generate loop body
        for (const auto& stmt : node->body) {
            visit(stmt.get(), output, indentLevel + 1);
        }
        
        indent(output, indentLevel);
        output << "} while (!(";
        visitExpression(node->condition.get(), output);
        output << "));\n";
    }

    std::string mapOperator(TokenType type) {
        switch(type) {
            case TokenType::PLUS: return "+";
            case TokenType::MINUS: return "-";
            case TokenType::MULTIPLY: return "*";
            case TokenType::DIVIDE: return "/";
            case TokenType::EQUAL: return "==";
            case TokenType::NOTEQUAL: return "!=";
            case TokenType::MORE: return ">";
            case TokenType::LESS: return "<";
            case TokenType::MOREOREQUAL: return ">=";
            case TokenType::LESSOREQUAL: return "<=";
            default: return "";
        }
    }
};


// Enhanced AST printing (for debugging)
void printAST(const std::unique_ptr<Node>& node, int indent = 0) {
    if (!node) return;
    std::string prefix(indent, ' ');
    
    if (auto* decl = dynamic_cast<DeclarationNode*>(node.get())) {
        std::cout << prefix << "DeclarationNode: " << decl->identifier.lexeme
                  << " (" << tokenTypeToString(decl->dataType.type) << ")\n";
        if (decl->arraySize1) {
            std::cout << prefix << "  Size1:\n";
            printAST(decl->arraySize1, indent + 4);
        }
        if (decl->arraySize2) {
            std::cout << prefix << "  Size2:\n";
            printAST(decl->arraySize2, indent + 4);
        }
        if (decl->initialValue) {
            std::cout << prefix << "  InitialValue:\n";
            printAST(decl->initialValue, indent + 4);
        }
    } else if (auto* output = dynamic_cast<OutputNode*>(node.get())) {
        std::cout << prefix << "OutputNode:\n";
        printAST(output->expression, indent + 2);
    } else if (auto* input = dynamic_cast<InputNode*>(node.get())) {
        std::cout << prefix << "InputNode: " << input->identifier.lexeme;
        if (input->index1 || input->index2) std::cout << " (with array indexing)";
        std::cout << "\n";
    } else if (auto* assign = dynamic_cast<AssignmentNode*>(node.get())) {
        std::cout << prefix << "AssignmentNode: " << assign->identifier.lexeme;
        if (assign->index1 || assign->index2) std::cout << " (with array indexing)";
        std::cout << "\n";
        printAST(assign->expression, indent + 2);
    } else if (auto* arrayAccess = dynamic_cast<ArrayAccessNode*>(node.get())) {
        std::cout << prefix << "ArrayAccessNode: " << arrayAccess->identifier.lexeme << "\n";
        if (arrayAccess->index1) {
            std::cout << prefix << "  Index1:\n";
            printAST(arrayAccess->index1, indent + 4);
        }
        if (arrayAccess->index2) {
            std::cout << prefix << "  Index2:\n";
            printAST(arrayAccess->index2, indent + 4);
        }
    } else if (auto* forLoop = dynamic_cast<ForLoopNode*>(node.get())) {  // <-- New
        std::cout << prefix << "ForLoopNode: " << forLoop->loopVariable.lexeme << "\n";
        std::cout << prefix << "  StartValue:\n";
        printAST(forLoop->startValue, indent + 4);
        std::cout << prefix << "  EndValue:\n";
        printAST(forLoop->endValue, indent + 4);
        std::cout << prefix << "  Body:\n";
        for (const auto& stmt : forLoop->body) {
            printAST(stmt, indent + 4);
        }
    } else if (auto* wait = dynamic_cast<WaitNode*>(node.get())) {
        std::cout << prefix << "WaitNode:\n";
        printAST(wait->seconds, indent + 2);
    } else if (auto* literal = dynamic_cast<LiteralNode*>(node.get())) {
        std::cout << prefix << "LiteralNode: " << literal->value.lexeme
                  << " (" << tokenTypeToString(literal->value.type) << ")\n";
    } else if (auto* identifier = dynamic_cast<IdentifierNode*>(node.get())) {
        std::cout << prefix << "IdentifierNode: " << identifier->identifier.lexeme << "\n";
    } else if (auto* unary = dynamic_cast<UnaryOpNode*>(node.get())) {
        std::cout << prefix << "UnaryOpNode: " << unary->op.lexeme << "\n";
        printAST(unary->operand, indent + 2);
    } else if (auto* binaryOp = dynamic_cast<BinaryOpNode*>(node.get())) {
        std::cout << prefix << "BinaryOpNode: " << binaryOp->op.lexeme << "\n";
        std::cout << prefix << "  Left:\n";
        printAST(binaryOp->left, indent + 4);
        std::cout << prefix << "  Right:\n";
        printAST(binaryOp->right, indent + 4);
    } else if (auto* ifStmt = dynamic_cast<IfStatementNode*>(node.get())) {
        std::cout << prefix << "IfStatementNode:\n";
        std::cout << prefix << "  Condition:\n";
        printAST(ifStmt->condition, indent + 4);
        std::cout << prefix << "  Then Body:\n";
        for (const auto& stmt : ifStmt->thenBody) { printAST(stmt, indent + 4); }
        if (!ifStmt->elseBody.empty()) {
            std::cout << prefix << "  Else Body:\n";
            for (const auto& stmt : ifStmt->elseBody) { printAST(stmt, indent + 4); }
        }
    } else if (auto* whileLoop = dynamic_cast<WhileLoopNode*>(node.get())) {
        std::cout << prefix << "WhileLoopNode:\n";
        std::cout << prefix << "  Condition:\n";
        printAST(whileLoop->condition, indent + 4);
        std::cout << prefix << "  Body:\n";
        for (const auto& stmt : whileLoop->body) {
            printAST(stmt, indent + 4);
        }
    } else if (auto* caseStmt = dynamic_cast<CaseOfNode*>(node.get())) {  // <-- Add this case
        std::cout << prefix << "CaseOfNode:\n";
        std::cout << prefix << "  Expression:\n";
        printAST(caseStmt->expression, indent + 4);
        std::cout << prefix << "  Cases:\n";
        for (const auto& caseItem : caseStmt->cases) {
            std::cout << prefix << "    Case Value:\n";
            printAST(caseItem->value, indent + 6);
            std::cout << prefix << "    Case Body:\n";
            for (const auto& stmt : caseItem->body) {
                printAST(stmt, indent + 6);
            }
        }
        if (!caseStmt->otherwiseBody.empty()) {
            std::cout << prefix << "  Otherwise Body:\n";
            for (const auto& stmt : caseStmt->otherwiseBody) {
                printAST(stmt, indent + 4);
            }
        }
    } else if (auto* repeatLoop = dynamic_cast<RepeatLoopNode*>(node.get())) {
        std::cout << prefix << "RepeatLoopNode:\n";
        std::cout << prefix << "  Body:\n";
        for (const auto& stmt : repeatLoop->body) {
            printAST(stmt, indent + 4);
        }
        std::cout << prefix << "  Until Condition:\n";
        printAST(repeatLoop->condition, indent + 4);
    }
}


// Enhanced compileSusi function with proper error handling
void compileSusi(const std::string& sourceCode) {
    try {
        std::cout << "Original Code:\n---\n" << sourceCode << "---\n\n";

        // Step 1: Tokenize the source code
        std::vector<Token> tokens = tokenize(sourceCode);
       
        // Step 2: Parse tokens and build the AST
        Parser parser(tokens);
        std::vector<std::unique_ptr<Node>> ast = parser.parse();
       
        // Step 3: Perform semantic analysis on the AST
        std::cout << "Performing semantic analysis...\n";
        SemanticAnalyzer analyzer;
        analyzer.analyze(ast);
        std::cout << "Semantic analysis passed!\n";
       
        // Step 4: Generate C++ code
        std::cout << "Generating C++ code...\n";
        CodeGenerator generator;
        std::string filename = "compiled_program.cpp";
       
        std::ofstream outFile(filename);
        if (!outFile) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        generator.generate(ast, outFile);
        outFile.close();
       
        std::cout << "C++ code generated successfully and saved to " << filename << "\n";
        std::cout << "You can now compile and run it with a C++ compiler (e.g., g++ " << filename << " -o program)\n";
       
    } catch (const std::exception& e) {
        std::cerr << "\nCOMPILATION ERROR:\n";
        std::cerr << "==================\n";
        std::cerr << e.what() << "\n\n";
        std::cerr << "Please fix the error and try again.\n";
    }
}


// Enhanced testing function that catches errors gracefully
void testCode(const std::string& code, const std::string& testName) {
    std::cout << "\n=== Testing: " << testName << " ===\n";
    std::cout << std::string(50, '-') << "\n";
    compileSusi(code);
    std::cout << std::string(50, '-') << "\n";
}