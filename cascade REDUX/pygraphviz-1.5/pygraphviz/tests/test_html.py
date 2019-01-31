# -*- coding: utf-8 -*-
from nose.tools import *
import pygraphviz as pgv
from os import linesep

long_html_string = '''<<TABLE BORDER=0>
  <TR>
      <TD> meow </TD>
  </TR>
  <TR>
      <TD>
        <TABLE>
          <TR>
          <TD align=left>Count</TD>
          <TD align=right> 4 </TD>
          </TR>
          <TR>
          <TD align=left>Earliest Run</TD>
          <TD align=right> yesterday </TD>
          </TR>
          <TR>
          <TD align=left>Latest Run</TD>
          <TD align=right> tomorrow </TD>
          </TR>
          <TR>
          <TD align=left>Avg Runtime</TD>
          <TD align=right> 4 seconds </TD>
          </TR>
          <TR>
          <TD align=left>Avg Failure Rate</TD>
          <TD align=right> 38.1% </TD>
          </TR>
        </TABLE>
      </TD>
  </TR>
</TABLE>>'''

def test_long_html_string():
    G = pgv.AGraph(label='<Hello<BR/>Graph>')
    G.add_node('a', label=long_html_string)
    s = G.add_subgraph('b', label='<Hello<BR/>Subgraph>')
    s.add_node('sa', label='<Hello<BR/>Subgraph Node b>')
    G.add_edge('a','b', label='<Hello<BR/>Edge>')
    assert_equal.__self__.maxDiff = None
    assert_equal(G.string().expandtabs(2),
"""strict graph {{
  graph [label=<Hello<BR/>Graph>];
  node [label="\\N"];
  {{
    graph [label=<Hello<BR/>Subgraph>];
    sa     [label=<Hello<BR/>Subgraph Node b>];
  }}
  a  [label={0}];
  a -- b   [label=<Hello<BR/>Edge>];
}}
""".format(long_html_string).replace('\n', linesep))

def test_html():
    G = pgv.AGraph(label='<Hello<BR/>Graph>')
    G.add_node('a', label='<Hello<BR/>Node>')
    s = G.add_subgraph('b', label='<Hello<BR/>Subgraph>')
    s.add_node('sa', label='<Hello<BR/>Subgraph Node b>')
    G.add_edge('a','b', label='<Hello<BR/>Edge>')
    assert_equal(G.string().expandtabs(2),
"""strict graph {
  graph [label=<Hello<BR/>Graph>];
  node [label="\\N"];
  {
    graph [label=<Hello<BR/>Subgraph>];
    sa     [label=<Hello<BR/>Subgraph Node b>];
  }
  a  [label=<Hello<BR/>Node>];
  a -- b   [label=<Hello<BR/>Edge>];
}
""".replace('\n', linesep))
