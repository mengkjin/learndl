def seperate_1_term_into_fml_terms(src_: str):
    src = src_.replace(' ', '').lower()
    assert len(src) > 0
    if '[' not in src:
        rtn = [None, src]
    else:
        assert src[-1] == ']', src_
        op_abbrev = ''
        for s in src:
            if s != '[' and s != ']':
                op_abbrev += s
            else:
                break
        assert src[len(op_abbrev)] == '['
        assert len(src) > len(op_abbrev) + 1
        input_expression = src[len(op_abbrev) + 1: -1]
        rtn = [op_abbrev, input_expression]
    return rtn


def split_expression_by_comma(expression_):
    term_list = list()
    n_left = 0
    term = ''
    for i in range(len(expression_)):
        s = expression_[i]
        if n_left == 0 and s == ',':
            term_list.append(term)
            term = ''
        elif s == "[":
            n_left += 1
            term += s
        elif s == "]":
            n_left -= 1
            term += s
        else:
            term += s
    if term != '':
        term_list.append(term)
    return term_list


def split_expression_by_semicolon(expression_):
    n_left = 0
    term1 = ''
    for i in range(len(expression_)):
        s = expression_[i]
        if n_left == 0 and s == ';':
            break
        elif s == '[':
            n_left += 1
            term1 += s
        elif s == ']':
            n_left -= 1
            term1 += s
        else:
            term1 += s
    if term1 == expression_:
        term2 = ''
    else:
        term2 = expression_[len(term1) + 1:]
    assert ';' not in term2
    return term1, term2