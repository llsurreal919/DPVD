% standalone.cls
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{standalone}[2012/09/15 v1.0 Making standalone files]

\DeclareOption{border}{\PassOptionsToPackage{border}{standalone}}
\DeclareOption{tikz}{\PassOptionsToPackage{tikz}{standalone}}
\DeclareOption{multi}{\PassOptionsToPackage{multi}{standalone}}

\ExecuteOptions{}
\ProcessOptions\relax

\LoadClass{article}
