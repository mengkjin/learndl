from collections import OrderedDict


class SectionLoader:
    def __init__(self, file_):
        self.m_file = file_
        self.m_lines = None
        self.m_sections = None
        self.load()

    def load(self):
        file = self.m_file
        with open(file, "r") as f:
            tmp = f.readlines()
        self.m_lines = list()
        for l in tmp:
            if l[0] != "#" and l != "\n":
                self.m_lines.append(l.replace("\n", ""))
        self._get_sections()

    def _get_sections(self):
        sections_start = OrderedDict()
        for i in range(len(self.m_lines)):
            if self.m_lines[i][0] == '[' and self.m_lines[i][-1] == ']':
                sections_start[self.m_lines[i]] = i
        sections_list = list(sections_start)
        assert len(self.m_lines) > sections_start[sections_list[-1]]
        sections_end = OrderedDict()
        for i in range(len(sections_list) - 1):
            sections_end[sections_list[i]] = sections_start[sections_list[i + 1]] - 1
        sections_end[sections_list[-1]] = len(self.m_lines)
        sections_lines = OrderedDict()
        for s in sections_list:
            sections_lines[s.replace("[", "").replace("]", "")] = self.m_lines[sections_start[s] + 1: sections_end[s] + 1]
        self.m_sections = sections_lines

    def items(self):
        return self.m_sections.items()

