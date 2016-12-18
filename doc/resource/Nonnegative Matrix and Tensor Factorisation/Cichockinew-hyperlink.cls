\def\currversion{November 2, 1999}

  %****************************************************************%
  %*                                                              *%
  %*     Prepared by Amy Hendrickson, TeXnology Inc.              *%
  %*     Copyright (C) 1996 by Amy Hendrickson, TeXnology Inc.,   *%
  %*     All rights reserved.                                     *%
  %*                                                              *%
  %*     TeXnology Inc.: 617 738-8029                             *%
  %*                     amyh@ai.mit.edu                          *%
  %*                                                              *%
  %****************************************************************%


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Macro Contents:

  %% Dimensions
  %% Global Parameters
  %% LaTeX2.09/LaTeX2e Switch, font family declarations
  %% Listing
  %% Title Pages, Preface, Introduction
  %% Part, Chapter, and Appendix Commands
  %% Counters
  %% Header Level Names
  %% Section Commands
  %% Verse, Quote, Extract
  %% Math
  %% Figure and Table Captions
  %% Table of Contents, List of Figures, LOT
  %% Bibliography, References
  %% Footnotes
  %% Running Heads
  %% Fonts for Specific Places
  %% Algorithm
  %% Glossary
  %% Index
  %% Draft line
  %% Two column, landscape, algorithm environment
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\typeout{^^J^^J
Wiley Book Style, Design No.: SD 001C^^J
^^J
Written by Amy Hendrickson, TeXnology, Inc., 10/96^^J
^^J
This version is dated \currversion^^J^^J}


%%% LaTeX2.09/LaTeX2e Switch  ====>>>
\newif\ifll
\expandafter\ifx\csname LaTeXe\endcsname\relax % LaTeX2.09 is used
\else% LaTeX2e is being used, so set ll true
\global\lltrue\fi

\ifll
%% (this may not be necessary)
\expandafter\ifx\csname mathunderscore\endcsname\relax
\def\mathunderscore{\leavevmode\kern.06em\vbox{\hrule width.3em}}
\def\mathellipsis{\mbox {$\m@th{\,.\,.\,.}\,$}}\fi
\global\newdimen\phspace
\else
\gdef\@plus{plus}
\gdef\@minus{minus}
\fi

%% Dimensions for both 2e and 2.09 ===>>>

\parindent=1pc
\parskip=0pt %plus .1pt

%\change by Cichocki textheight from 45pc to 47.5pc
\textheight = 48pc
%\textheight = 45pc
%change by Cichocki textwidth from 28pc to 33pc
\textwidth 134mm %%31.5pc
%\textwidth 28pc

\topmargin.75in
\headheight 12pt
\headsep 18pt
\topskip  10pt
\footskip 18pt

\columnsep 10pt
\columnseprule 0pt

\footnotesep 6.65pt
\skip\footins 24pt plus 4pt minus 2pt

\floatsep 12pt plus 2pt minus 2pt
\textfloatsep 20pt plus 2pt minus 4pt
\intextsep 12pt plus 2pt minus 2pt

\dblfloatsep 12pt plus 2pt minus 2pt
\dbltextfloatsep 20pt plus 2pt minus 4pt

\ifll\else
\global\@maxsep 20pt
\global\@dblmaxsep 20pt
\fi

\@fptop 0pt plus 1fil
\@fpsep 8pt plus 2fil
\@fpbot 0pt plus 1fil
\@dblfptop 0pt plus 1fil
\@dblfpsep 8pt plus 2fil
\@dblfpbot 0pt plus 1fil

\maxdepth=4pt %%

%change by Cichocki oddsidemargin from .5 to .1 
% and evensidemarigin from 1.5 to 0.7
%\oddsidemargin.05in
\oddsidemargin.55in
\evensidemargin.55in
%\evensidemargin  1.5in
\marginparwidth .75in

\if@twocolumn
 \setlength\marginparsep {10\p@}
\else
  \setlength\marginparsep{7\p@}
\fi
\setlength\marginparpush{5\p@}

\setlength\partopsep{2\p@ \@plus 1\p@ \@minus 1\p@}

%%%
\setcounter{topnumber}{10}
\def\topfraction{.9}
\setcounter{bottomnumber}{10}
\def\bottomfraction{.9}
\setcounter{totalnumber}{20}
\def\textfraction{.1}
\def\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\def\dbltopfraction{.7}
\def\dblfloatpagefraction{.5}
%%%

\setlength\arraycolsep{5\p@}
\tabcolsep 6pt

\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}

\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}

%%%%%%%%%%%%%%% <<== end dimensions

%%% Global parameters ==>>

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}

%% to make left and right page position differently.
\@twosidetrue  \@mparswitchtrue
\@twocolumnfalse

%% 2e ==>>>
\newif\if@openbib
%\@openbibtrue
\@openbibfalse
\newif\if@openright
\newif\if@mainmatter
\newif\if@restonecol
\newif\if@titlepage
\newif\ifdraft

%% Start new chapter on right side
\newif\if@openright
\@openrighttrue
%% <<== end 2e

\newcommand{\contentsname}{Contents}
\newcommand{\listfigurename}{List of Figures}
\newcommand{\listtablename}{List of Tables}
\newcommand{\bibname}{Bibliography}
\newcommand{\indexname}{Index}
\newcommand{\figurename}{Fig.}
\newcommand{\tablename}{Table}
\newcommand{\partname}{Part}
\newcommand{\chaptername}{Chapter}
\newcommand{\appendixname}{Appendix}

%%% <== end global parameters


\font\egtbf=cmbx8
\font\sevbf=cmbx7
\font\fivbf=cmbx5

\ifll \let\dooptions\ProcessOptions
\else
\let\dooptions\@options\fi
\dooptions

\ifll
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Cichockinew}
%\ProvidesClass{w-book}
              [1996/10/08]

\newcommand{\@ptsize}{}

\renewcommand{\normalsize}{%
\ifdraft
   \@setfontsize\normalsize\@xpt\@xviipt
\else
   \@setfontsize\normalsize\@xpt\@xiipt
\fi
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
\normalsize
\newcommand{\small}{%
\ifdraft
   \@setfontsize\small\@ixpt{17}%
\else
   \@setfontsize\small\@ixpt{11}%
\fi
   \abovedisplayskip 8.5\p@ %\@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ %\@plus2\p@
   \belowdisplayshortskip 4\p@ %\@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand{\footnotesize}{%
\ifdraft
   \@setfontsize\footnotesize\@viiipt{17}%
\else
   \@setfontsize\footnotesize\@viiipt{9.5}%
\fi
   \abovedisplayskip 6\p@ %\@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ %\@plus\p@
   \belowdisplayshortskip 3\p@ %\@plus\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 3\p@ \@plus\p@ \@minus\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand{\xscriptsize}{\@setfontsize\scriptsize\@vipt\@viipt}
\newcommand{\scriptsize}{\@setfontsize\scriptsize\@viipt\@viiipt}
\newcommand{\tiny}{\@setfontsize\tiny\@vpt\@vipt}
\newcommand{\large}{\@setfontsize\large\@xiipt{14}}
\newcommand{\Large}{\@setfontsize\Large\@xivpt{18}}
\newcommand{\LARGE}{\@setfontsize\LARGE\@xviipt{22}}
\newcommand{\huge}{\@setfontsize\huge\@xxpt{25}}
\newcommand{\Huge}{\@setfontsize\Huge\@xxvpt{30}}


\@maxdepth\maxdepth
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand{\cal}{\@fontswitch{\relax}{\mathcal}}
\DeclareRobustCommand{\mit}{\@fontswitch{\relax}{\mathnormal}}

\else %% LaTeX2.09
%% To get bold super and sub-scripts in the right size:
\gdef\xpt{\textfont\z@\tenrm
  \scriptfont\z@\sevrm \scriptscriptfont\z@\fivrm
\textfont\@ne\tenmi \scriptfont\@ne\sevmi \scriptscriptfont\@ne\fivmi
\textfont\tw@\tensy \scriptfont\tw@\sevsy \scriptscriptfont\tw@\fivsy
\textfont\thr@@\tenex \scriptfont\thr@@\tenex \scriptscriptfont\thr@@\tenex
\def\unboldmath{\everymath{}\everydisplay{}\@nomath\unboldmath
          \textfont\@ne\tenmi
          \textfont\tw@\tensy \textfont\lyfam\tenly
          \@boldfalse}\@boldfalse
\def\boldmath{\@ifundefined{tenmib}{\global\font\tenmib\@mbi
   \global\font\tensyb\@mbsy
   \global\font\tenlyb\@lasyb\relax\@addfontinfo\@xpt
   {\def\boldmath{\everymath{\mit}\everydisplay{\mit}\@prtct\@nomathbold
        \textfont\@ne\tenmib \textfont\tw@\tensyb
        \textfont\lyfam\tenlyb \@prtct\@boldtrue}}}{}\@xpt\boldmath}%
\def\prm{\fam\z@\tenrm}%
\def\pit{\fam\itfam\tenit}\textfont\itfam\tenit \scriptfont\itfam\sevit
    \scriptscriptfont\itfam\sevit
\def\psl{\fam\slfam\tensl}\textfont\slfam\tensl
     \scriptfont\slfam\tensl \scriptscriptfont\slfam\tensl
\def\pbf{\fam\bffam\tenbf}\textfont\bffam\tenbf
    \scriptfont\bffam\sevbf \scriptscriptfont\bffam\fivbf
\def\ptt{\fam\ttfam\tentt}\textfont\ttfam\tentt
    \scriptfont\ttfam\tentt \scriptscriptfont\ttfam\tentt
\def\psf{\fam\sffam\tensf}\textfont\sffam\tensf
    \scriptfont\sffam\tensf \scriptscriptfont\sffam\tensf
\def\psc{\@getfont\psc\scfam\@xpt{\@mcsc}}%
\def\ly{\fam\lyfam\tenly}\textfont\lyfam\tenly
   \scriptfont\lyfam\sevly \scriptscriptfont\lyfam\fivly
\@setstrut \rm}
%
\gdef\@normalsize{\ifdraft
\@setsize\normalsize{17pt}\xpt\@xpt
\else
\@setsize\normalsize{12pt}\xpt\@xpt
\fi
\abovedisplayskip 10\p@ plus2\p@ minus5\p@
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip  \z@ plus3\p@
\belowdisplayshortskip  6\p@ plus3\p@ minus3\p@
\let\@listi\@listI}
\gdef\small{\ifdraft
\@setsize\small{17pt}\ixpt\@ixpt
\else
\@setsize\small{11pt}\ixpt\@ixpt
\fi
\abovedisplayskip 8.5\p@ %plus3\p@ minus4\p@
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ %plus2\p@
\belowdisplayshortskip 4\p@ %plus2\p@ minus2\p@
\def\@listi{\leftmargin\leftmargini
\topsep 4\p@ plus2\p@ minus2\p@\parsep 2\p@ plus\p@ minus\p@
\itemsep \parsep}}
\gdef\footnotesize{\ifdraft
\@setsize\footnotesize{17pt}\viiipt\@viiipt
\else
\@setsize\footnotesize{9.5pt}\viiipt\@viiipt
\fi
\abovedisplayskip 6\p@ %plus2\p@ minus4\p@
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ %plus\p@
\belowdisplayshortskip 3\p@ %plus\p@ minus2\p@
\def\@listi{\leftmargin\leftmargini
\topsep 3\p@ plus\p@ minus\p@\parsep 2\p@ plus\p@ minus\p@
\itemsep \parsep}}
%
\gdef\scriptsize{\@setsize\scriptsize{8pt}\viipt\@viipt}
\gdef\xscriptsize{\@setsize\scriptsize{8pt}\viipt\@viipt}
\gdef\tiny{\@setsize\tiny{6pt}\vpt\@vpt}
\gdef\large{\@setsize\large{14pt}\xiipt\@xiipt}
\gdef\Large{\@setsize\Large{18pt}\xivpt\@xivpt}
\gdef\LARGE{\@setsize\LARGE{22pt}\xviipt\@xviipt}
\gdef\huge{\@setsize\huge{25pt}\xxpt\@xxpt}
\gdef\Huge{\@setsize\Huge{30pt}\xxvpt\@xxvpt}
%
\gdef\bfseries{\bf}
\fi

\lineskip 1pt
\normallineskip 1pt
\def\baselinestretch{1}

\normalsize
%% <<== end font info


%% Listing, same as 2.09 ==>>

\def\@listI{\leftmargin\leftmargini
            \parsep 4\p@ \@plus2\p@ \@minus\p@
            \topsep 8\p@ \@plus2\p@ \@minus4\p@
            \itemsep4\p@ \@plus2\p@ \@minus\p@}
\let\@listi\@listI
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep    4\p@ \@plus2\p@ \@minus\p@
              \parsep    2\p@ \@plus\p@  \@minus\p@
              \itemsep   \parsep}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2\p@ \@plus\p@\@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   \topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}

\leftmargini 25pt
\leftmarginii 22pt
\leftmarginiii 18.7pt
\leftmarginiv 17pt
\leftmarginv 10pt
\leftmarginvi 10pt

\leftmargin\leftmargini
\labelsep 5pt
\labelwidth\leftmargini\advance\labelwidth-\labelsep

\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength\leftmargin    {\leftmargini}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}

\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty

\renewcommand{\theenumi}{\arabic{enumi}}
\renewcommand{\theenumii}{\alph{enumii}}
\renewcommand{\theenumiii}{\roman{enumiii}}
\renewcommand{\theenumiv}{\Alph{enumiv}}
\newcommand{\labelenumi}{\theenumi.}
\newcommand{\labelenumii}{(\theenumii)}
\newcommand{\labelenumiii}{\theenumiii.}
\newcommand{\labelenumiv}{\theenumiv.}
\renewcommand{\p@enumii}{\theenumi}
\renewcommand{\p@enumiii}{\theenumi(\theenumii)}
\renewcommand{\p@enumiv}{\p@enumiii\theenumiii}
\newcommand{\labelitemi}{$\m@th\bullet$}
\newcommand{\labelitemii}{\normalfont\bfseries --}
\newcommand{\labelitemiii}{$\m@th\ast$}
\newcommand{\labelitemiv}{$\m@th\cdot$}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}

\ifll
\newcommand*{\descriptionlabel}[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\else
\gdef\descriptionlabel#1{\hspace\labelsep \bf #1}\fi

%%% <<=== end of listing commands

%% Title Pages, Preface, Introduction==>>

\def\preface{\startonoddpage
\chapter*{Preface}
\global\titletrue
\markboth{PREFACE}{PREFACE}
\addtocontents{toc}{\vskip-8pt}
\addcontentsline{toc}{chapter}{\protect\numberline{\ }Preface}}

\def\endpreface{\newpage\markboth{}{}}
\def\prefaceauthor#1{\def\pone{#1}\vskip14pt\hbox to\textwidth{\hfill
\expandafter\fakesc\pone*}}

\def\where#1{\vskip4pt\noindent{\wherefont #1}}

\def\acronyms{\startonoddpage
\global\titletrue
\chapter*{Acronyms}
\markboth{Acronyms}{Acronyms}
\addtocontents{toc}{\vskip-8pt}
\addcontentsline{toc}{chapter}{\protect\numberline{\ }Acronyms}}

\def\acro#1#2{\vskip4pt\hbox to\textwidth{\large
\hbox to5pc{#1\hfill}\vtop{\advance\hsize by
-5pc\raggedright\noindent#2}}}

\def\introduction{\startonoddpage
\global\titletrue
\chapter*{Introduction}
\markboth{INTRODUCTION}{INTRODUCTION}
\renewcommand{\thechapter}{I}
\addtocontents{toc}{\vskip-8pt}
\addcontentsline{toc}{chapter}{\protect\numberline{\ }Introduction}}

\def\acknowledgements{\startonoddpage
\global\titletrue
\chapter*{Acknowledgments}
\addtocontents{toc}{\vskip-8pt}
\addcontentsline{toc}{chapter}{\numberline{\ }Acknowledgments}}

\def\authorinitials#1{\vskip12pt
\hbox to\textwidth{\hfill#1}\newpage}
%% from 2e
\newcommand{\frontmatter}{\cleardoublepage
            \@mainmatterfalse\pagenumbering{roman}}
\newcommand{\mainmatter}{\cleardoublepage
       \@mainmattertrue\pagenumbering{arabic}}
\newcommand{\backmatter}{\if@openright\cleardoublepage\else\clearpage\fi
      \@mainmatterfalse}
%%

\newif\iftitle

\def\booktitle#1{\def\thetitle{#1}}
\def\subtitle#1{\def\thesubtitle{#1}}
\def\author#1{\def\theauthor{#1}}
\def\editionstatement#1{\def\theeditionstatement{#1}}

\booktitle{}
\subtitle{}
\editionstatement{}

\def\halftitlepage{\thispagestyle{empty}
  \vspace*{48pt}%
  {\parindent \z@
    \interlinepenalty\@M
\LARGE
\hyphenpenalty10000
\everypar={\hskip0pt plus 1fil\relax}
\parfillskip=0pt
\bfseries\halftitlefont
\def\\ {\vskip1sp}
\thetitle\vskip 1sp
}
\newpage
}

\def\titlepage{\ifodd\c@page\else\thispagestyle{empty}\null\newpage\fi
\thispagestyle{empty}
\vbox to\vsize{\vspace*{64pt}%
\parindent \z@
   \interlinepenalty\@M
\Huge
\hyphenpenalty10000
\everypar={\hskip0pt plus 1fil\relax}
\parfillskip=0pt
\bfseries\titlefont
\def\\ {\vskip1sp}
\thetitle
\ifx\subtitle\empty\else
\vskip2pc
{\baselineskip22pt
\subtitlefont \thesubtitle\vskip1sp}\fi
\ifx\theeditionstatement\empty\else
\vskip48pt
{\editionfont \theeditionstatement}\fi
\vskip6.5pc
{\authorfont \theauthor}
\vfill
{\imprintfont\parskip=0pt \baselineskip=16pt
% A Wiley-Interscience Publication
%CCC
%{\wileyfont JOHN WILEY \& SONS, LTD.}

%Chichester \dotspace New York\dotspace Weinheim%
%\dotspace Brisbane\dotspace Singapore\dotspace Toronto}}
Tokyo}}
\newpage
\null
\setcounter{page}{4}
\thispagestyle{empty}
\newpage
}



\def\dotspace{\hskip5pt/\hskip5pt}

%% end Title Pages, Preface

%% Part, Chapter, and Appendix Commands ===>>>

\def\chaptermark#1{}

\newcommand{\part}{\startonoddpage
                 \thispagestyle{empty}%
                     \@tempswafalse
                 \secdef\@part\@spart}

\def\@part[#1]#2{%
\ifnum\c@part=0 \global\@mainmattertrue
\pagenumbering{arabic}\fi
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}
  \vspace*{70pt}%
  {\parindent \z@
    \interlinepenalty\@M
\Large
\hbox to\textwidth{\hfill
\hbox{\partnumberfont
\partname~\Roman{part}}\hskip4pc}
\vskip-3pt
\hbox to\textwidth{\hfill\hbox to15pc{\hrulefill}}
\vskip12pt
\vtop to8.5pc{\leftskip=0pt plus 1fil
\hyphenpenalty10000
\everypar={\hskip0pt plus 1fil\relax}
\parfillskip=0pt
\large\baselineskip=32pt\bfseries\partfont
\def\\ {\vskip1sp\relax}
#2\vskip 1sp
\vfill}}}



\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \reset@font
     \Huge \bfseries #1\par}%
    \@endpart}
\def\@endpart{\vfill\newpage
              \if@twoside
                \hbox{}%
                \thispagestyle{empty}%
                \newpage
             \fi}


\def\and{\lowercase{and}\ }

\def\eatone#1{}
\def\oldauthor[#1]#2{{\ifnum\c@chapter<1
\vskip-2pc
\else
\vskip2pc
\ifnum\c@chapter>9\leftskip=4pc\else
\leftskip=3pc\fi\fi\let\\ \newline{#2}\vskip1sp}
{\let\\ \space \let\footnote\eatone\xdef\totoc{\tocauthorfont #1}}
\addtocontents{toc}{\vskip-2.5pt{\string\noindent\hskip24pt\totoc}}
\addtocontents{toc}{\vskip6pt}
}


\newif\iffirstprologue
\def\prologue#1#2{\iffirstprologue\vskip4pt
\global\firstprologuefalse\else
\vskip12pt\fi
{\leftskip=1pc
\parindent=0pt \small
\let\\ \newline#1\vskip1sp
\hbox to\textwidth{\hfill---#2}
}}


\def\chapter{
%%\ifnum\value{chapter}>\value{foo}
%%    \refixappendix
%%\fi
\global\footnum=0
\global\probcount=0
\global\c@chapapp=0
\renewcommand{\thechapter}{\arabic{chapter}}
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}

\def\@chapter[#1]#2{\startonoddpage
\global\titletrue
%% no floats on chapter title page
\global\@topnum\z@
\global\@botnum\z@
%%
\ifnum\c@chapter=0 \setcounter{page}{1}%
\global\@mainmattertrue% <<===
\fi
%
\gdef\thepage{\csname @arabic\endcsname\c@page}
%
\refstepcounter{chapter}% <<==== moved this here instead of below mainmatter
\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                      \@makechapterhead{{\def\\ {\vskip1sp}#2}}%
                      \@afterheading
                    }

\def\@makechapterhead#1{%
  \vspace*{70pt}%
  {\parindent \z@
    \interlinepenalty\@M
\Large
\hbox to\textwidth{\hfill
\hbox{\chaptercolor\chapternumberfont
\the\c@chapter}\hskip4pc}
\vskip-4pt
\chaptercolor\hbox to\textwidth{\hfill\hbox to15pc{\hrulefill}}
\vskip7pt
\vtop to8.5pc{\leftskip=0pt plus 1fil
\hyphenpenalty10000
\everypar={\hskip0pt plus 1fil\relax}
\parfillskip=0pt
\huge\bfseries\chaptercolor\chaptertitlefont
#1\vskip 1sp
\vfill}}
}

\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi\global\titletrue\relax}

\def\@makeschapterhead#1{%
  \vspace*{10.5pc}%
\hbox to\textwidth{\hfill\Large\bfseries\chaptercolor\chaptertitlefont #1}\nobreak
\vskip8pc}

\let\title\chapter

%% This code is used to fix appendix hyperlinks - Anh Huy Phan, 03/2009
\newcounter{foo}
\newcounter{chaptertrue}
\setcounter{foo}{100}
\newcommand\fixappendix{
\setcounter{chapter}{\value{chapter}+\value{foo}}}
\newcommand\refixappendix{\setcounter{chapter}{\value{chaptertrue}}}

\newcount\c@chapapp
\def\chapappendix#1{\par
\global\advance\c@chapapp by 1
\ifnum\value{chapter}>\value{foo}
    \refixappendix
\fi
\setcounter{chaptertrue}{\value{chapter}}
\fixappendix

  \setcounter{section}{0}%
  \setcounter{figure}{0}%
  \setcounter{table}{0}%
  \setcounter{equation}{0}%
  \renewcommand{\@chapapp}{\appendixname}%
  \renewcommand{\thechapter}{\Alph{chapapp}}
\def\one{#1}\ifx\one\empty
\section*{Appendix \arabic{chaptertrue}.\Alph{chapapp}}
\markright{APPENDIX}
\addcontentsline{toc}{section}{\protect\numberline{}Appendix}
\else
\section*{Appendix \arabic{chaptertrue}.\Alph{chapapp}. #1}
\markright{APPENDIX}
\addcontentsline{toc}{section}{\protect\numberline{}Appendix #1}
\fi
}

\newif\iffirstappendix
\global\firstappendixtrue

\newcount\c@appfootnote


\def\lookatapp{\if\next[\let\go\yappendix\else
\let\go\xappendix\fi\go}
\def\appendix{\futurelet\next\lookatapp}

\def\appthanks#1{*\gdef\realthanks{#1}}

\def\yappendix[#1]#2{\startonoddpage
\global\titletrue
\global\@mainmattertrue% <<===
\gdef \thepage {\csname @arabic\endcsname \c@page }
%% To fix includeonly numbering bug: ====>>
\if@partsw\else
\iffirstappendix\setcounter{chapter}{0}\fi
\global\firstappendixfalse\fi%
\refstepcounter{chapter}
% <<====  end bug fix
  \setcounter{section}{0}%
  \setcounter{figure}{0}%
  \setcounter{table}{0}%
  \setcounter{equation}{0}%
  \renewcommand{\@chapapp}{\appendixname}%
\def\two{#2}\ifx\two\empty
\chapter*{Appendix \Alph{chapter}}
\renewcommand{\thechapter}{\Alph{chapter}}
\addcontentsline{toc}{chapter}{ Appendix \thechapter}
\else
\chapter*{\vtop{\let\thanks\appthanks
\let\\\break \baselineskip=30pt \parfillskip0pt
\leftskip=0pt plus 1fil
 Appendix \Alph{chapter}\vskip4pt
 #2}}
\renewcommand{\thechapter}{\Alph{chapter}}
\addcontentsline{toc}{chapter}{ Appendix \thechapter\space #1}
\fi
\def\one{#1}
\ifx\one\empty
\chaptermark{Appendix \thechapter}
\else
\chaptermark{#1}
\fi
\xdef\@currentlabel{\thechapter}
\ifx\realthanks\xrelax\else
\xdef\dofoot{\realthanks}
\fi
{\def\thefootnote{*}
\footnotetext{\dofoot}}
\gdef\realthanks{\relax}
}

\let\dofoot\relax
\gdef\realthanks{\relax}
\def\xrelax{\relax}
\def\xappendix#1{\startonoddpage
\global\titletrue
\global\@mainmattertrue% <<===
\gdef \thepage {\csname @arabic\endcsname \c@page }
%% To fix includeonly numbering bug: ====>>
\if@partsw\else
\iffirstappendix\setcounter{chapter}{0}\fi
\global\firstappendixfalse\fi%
\refstepcounter{chapter}
% <<====  end bug fix
  \setcounter{section}{0}%
  \setcounter{figure}{0}%
  \setcounter{table}{0}%
  \setcounter{equation}{0}%
  \renewcommand{\@chapapp}{\appendixname}%
\def\one{#1}\ifx\one\empty
\chapter*{Appendix \Alph{chapter}}
\renewcommand{\thechapter}{\Alph{chapter}}
\addcontentsline{toc}{chapter}{ Appendix \thechapter}
\else
\chapter*{\vtop{\let\thanks\appthanks
\let\\\break \baselineskip=30pt \parfillskip0pt
\leftskip=0pt plus 1fil
 Appendix \Alph{chapter}\vskip4pt
 #1}}
\renewcommand{\thechapter}{\Alph{chapter}}
\addcontentsline{toc}{chapter}{ Appendix \thechapter\space #1}
\fi
\def\one{#1}
\ifx\one\empty
\chaptermark{Appendix \thechapter}
\else
\chaptermark{#1}
\fi
\xdef\@currentlabel{\thechapter}
}

%%% <<=== end Chapter, Part, and Appendix Commands


%%% Counters ===>>
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]

%%% <=== end counters

%%% Header Level Names ==>>
\renewcommand{\thepart}         {\Roman{part}}
\renewcommand{\thechapter}      {\arabic{chapter}}
\renewcommand{\thesection}      {\thechapter.\arabic{section}}
\renewcommand{\thesubsection}   {\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection .\arabic{subsubsection}}
\renewcommand{\theparagraph}    {\thesubsubsection.\arabic{paragraph}}
\renewcommand{\thesubparagraph} {\theparagraph.\arabic{subparagraph}}
\newcommand{\@chapapp}{\chaptername}

%%% <<== End Header Level Names


%%% Section Commands ====>>>>

% \@startsection {NAME}{LEVEL}{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}
%            optional * [ALTHEADING]{HEADING}
%    Generic command to start a section.
%    NAME       : e.g., 'subsection'
%    LEVEL      : a number, denoting depth of section -- e.g., chapter=1,
%                 section = 2, etc.
%    INDENT     : Indentation of heading from left margin
%    BEFORESKIP : Absolute value = skip to leave above the heading.
%                 If negative, then paragraph indent of text following
%                 heading is suppressed.
%    AFTERSKIP  : if positive, then skip to leave below heading, else
%                 negative of skip to leave to right of run-in heading.
%    STYLE      : commands to set style
%  If '*' missing, then increments the counter.  If it is present, then
%  there should be no [ALTHEADING] argument.
%  Uses the counter 'secnumdepth' whose value is the highest section
%  level that is to be numbered.


\def\@sect#1#2#3#4#5#6[#7]#8{\let\dolabelnow\relax%
\ifnum #2>\c@secnumdepth
     \let\@svsec\@empty\else
     \refstepcounter{#1}\edef\@svsec{\csname the#1\endcsname\hskip 1em }\fi
     \@tempskipa #5\relax
\ifdim \@tempskipa>\z@
%
        \begingroup#6\relax
          \@hangfrom{\hskip #3\relax\@svsec}{\interlinepenalty \@M
     \hyphenpenalty=10000 % No hyphenation in section heads
     \raggedright
     %% make section head uppercase
     \ifnum#2=1{\def\label##1{\gdef\dolabelnow{\savelabel{##1}}}%
     \global\setbox0=\hbox{\def\\ {\relax}#8}}%
     \uppercase{\fi%
     \def\label##1{}%
     \def\\ {\hfill\break}%
     #8\ifnum#2=1}\fi\par}%%
     \endgroup%
     \csname #1mark\endcsname{#7} %
%
     {%<===
     \def\\ { }%
    \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth\else%
                       \protect\numberline{\csname the#1\endcsname}\fi
                    #7 }%<===
    }%<===
%
\else
        \def\@svsechd{#6\hskip #3\relax
                 \@svsec #8\csname #1mark\endcsname
                   {#7}\addcontentsline
                     {toc}{#1}{\ifnum #2>\c@secnumdepth \else
                              \protect\numberline{\csname the#1\endcsname}\fi
                       #7}}\fi\@xsect{#5}
\dolabelnow %% to make label not be uppercase.
}




\newcommand{\section}{\@startsection {section}{1}{\z@}%
                                   {-22pt %plus 1pt minus 6pt
}%
                                   {13pt}%
                                   {\reset@font\large\bfseries\boldmath\sectioncolor\sectionfont}}
\newcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
                                     {-16pt %plus 1pt minus 3pt
}%
                                     {8pt}%
                                 {\reset@font\large\bfseries\boldmath\subsectioncolor\subsectionfont}}
\newcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
                                     {-12pt %plus 1pt minus 6pt
}%
                                     {-1em}
                           {\reset@font\normalsize\bfseries\boldmath\subsubsectioncolor\subsubsectionfont}}
\newcommand{\paragraph}{\@startsection{paragraph}{4}{\z@}%
                                    {-12pt}%
                                    {-1em}%
                               {\reset@font\normalsize\bfseries\paragraphcolor\paragraphfont}}
\newcommand{\subparagraph}{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\reset@font\normalsize\bfseries}}

%%% <<=== end section commands


%%% Verse, Quote, Extract ==>>

\newenvironment{verse}
               {\let\\=\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item[]}
               {\endlist}
\newenvironment{quotation}
               {\small
\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item[]}
               {\endlist}
\newenvironment{quote}
               {\small
\list{}{\rightmargin\leftmargin}%
                \item[]}
               {\endlist}
\let\extract\quotation
\let\endextract\endquotation
%% <<== end Verse, Quote, Extract


%%% Math ===>>>

\@addtoreset{equation}{chapter}
\renewcommand{\theequation}{\thechapter.\arabic{equation}}

%%% <<<=== end Math


%%% Figure and Table Captions ==>>>
\newcounter{figure}[chapter]
\renewcommand{\thefigure}{\thechapter.%
\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{{\figurecolor\figurename~\thefigure}\captioncolor}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand{\thetable}{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{{\tablecolor\tablename~\thetable}\captioncolor}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
   
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}

\def\xfigure{figure}
\long\def\@makecaption#1#2{%
\vskip\abovecaptionskip
{\small
 \sbox\@tempboxa{{\fignumfont#1\quad \ifx\@captype\xfigure\figtextfont\else
\tabtextfont\fi\let\newline\ #2}}%
  \ifdim \wd\@tempboxa >.9\hsize
\fignumfont #1\quad
\ifx\@captype\xfigure\figtextfont\else
\tabtextfont\fi#2\par
  \else
    \global \@minipagefalse
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
\ifx\@captype\xfigure
  \vskip\belowcaptionskip
\else
\vskip3pt\fi
}}


\newif\ifsettotop

\ifll
\def\@xfloat#1[#2]{%
\def \@captype {#1}\def \@fps {#2}\@onelevel@sanitize \@fps
\def \reserved@b {!}\ifx \reserved@b \@fps \@fpsadddefault \else
\ifx \@fps \@empty \@fpsadddefault \fi \fi
\ifhmode \@bsphack \@floatpenalty -\@Mii \else \@floatpenalty -\@Miii \fi
\ifinner \@parmoderr \@floatpenalty \z@
\else \@next \@currbox \@freelist {\@tempcnta \sixt@@n
\expandafter \@tfor \expandafter \reserved@a \expandafter :\expandafter
=\@fps \do {\if \reserved@a h\ifodd \@tempcnta \else \advance \@tempcnta \@ne
\fi \fi \if \reserved@a t\@setfpsbit \tw@ \global\settotoptrue\fi
\if \reserved@a b\@setfpsbit 4\fi
\if \reserved@a p\@setfpsbit 8\fi
\if \reserved@a !\ifnum \@tempcnta >15
\advance\@tempcnta -\sixt@@n\relax\fi \fi }\@tempcntb
\csname ftype@\@captype \endcsname \multiply \@tempcntb \@xxxii \advance
\@tempcnta \@tempcntb \global \count \@currbox \@tempcnta }\@fltovf \fi
\global \setbox \@currbox \color@vbox \normalcolor \vbox \bgroup
\ifx\@captype\xtable
\ifsettotop
\vglue-6pt
\fi\fi\global\settotopfalse
\hsize \columnwidth \@parboxrestore \@floatboxreset}
\else
\def\@xfloat#1[#2]{%
\ifhmode \@bsphack \@floatpenalty -\@Mii \else \@floatpenalty -\@Miii
\fi
\def \@captype {#1}\ifinner \@parmoderr \@floatpenalty \z@ \else
\@next \@currbox \@freelist {\@tempcnta \csname ftype@#1\endcsname
\multiply \@tempcnta \@xxxii \advance \@tempcnta \sixt@@n \@tfor \@tempa :=#2
\do {\if \@tempa h\advance \@tempcnta \@ne \fi
\if \@tempa t\advance \@tempcnta \tw@  \global\settotoptrue\fi
\if \@tempa b\advance \@tempcnta 4\relax \fi
\if \@tempa p\advance \@tempcnta 8\relax \fi }%
\global \count \@currbox \@tempcnta }\@fltovf \fi
\global \setbox \@currbox \vbox \bgroup
\ifx\@captype\xtable
\ifsettotop
\vglue-6pt
\fi\fi\global\settotopfalse
\hsize \columnwidth \@parboxrestore}
\fi

\let\savetable\table

\def\table{\small\savetable}


%%% code to get text in tables to extend all the way to left and right
%%% of table

\def\xtable{table}
\def\@array[#1]#2{\setbox\@arstrutbox=\hbox{\vrule
     height\arraystretch \ht\strutbox
     depth\arraystretch \dp\strutbox
     width\z@}\@mkpream{#2}\edef\@preamble{\halign \noexpand\@halignto
\bgroup%
\tabskip\z@\@arstrut\@preamble
\ifx\@captype\xtable\hskip-\tabcolsep\fi%% <==== Changed
\tabskip\z@ \cr}%
\let\@startpbox\@@startpbox \let\@endpbox\@@endpbox%
  \if #1t\vtop \else \if#1b\vbox \else \vcenter \fi\fi%
  \bgroup\let\par\relax%
  \let\@sharp##\let\protect\relax \lineskip\z@\baselineskip\z@\@preamble}

%% new
\def\new@tabacol{\edef\@preamble{\@preamble\hskip0pt}}

\def\@tabclassz{\ifcase \@lastchclass\@acolampacol%
\or \@ampacol \or
   \or \or \@addamp \or \@acolampacol\or \@firstampfalse
\ifx\@captype\xtable \new@tabacol\else\@tabacol \fi%
\fi%
\edef\@preamble{\@preamble%
  \ifcase \@chnum%
     \hfil\ignorespaces\@sharp\unskip\hfil%
     \or \ignorespaces\@sharp\unskip\hfil%
     \or \hfil\hskip\z@ \ignorespaces\@sharp\unskip\fi}}

\let\savehline\hline
\def\hline{\noalign{\vskip3pt}\savehline\noalign{\vskip3pt}}

\def\tablenotes{\vskip2pt\footnotesize}
\let\endtablenotes\relax
%%% <<=== end Figure and Table Captions


%%% Table of Contents, List of Figures, LOT ===>>

\newcommand{\@pnumwidth}{1.55em}
\newcommand{\@tocrmarg} {2.55em  plus .5in}
\newcommand{\@dotsep}{4.5}

\def\startonoddpage{\clearpage
\ifodd\c@page\else\null\thispagestyle{empty}\newpage\fi}

\newcommand{\tableofcontents}{%
\startonoddpage
\global\titletrue
    \chapter*{\contentsname
        \@mkboth{\uppercase{\contentsname}}{\uppercase{\contentsname}}}%
\vskip-9pt
\bgroup\parskip=2pt
{ \tocfonts 
\hyphenpenalty=10000
    \@starttoc{toc}%
}
\egroup
    \if@restonecol\twocolumn\fi

\newpage
\markboth{}{}
    }

\newcommand{\listoftables}{%
\startonoddpage
\global\titletrue
    \chapter*{List of Tables
        \@mkboth{\uppercase{\contentsname}}{\uppercase{\contentsname}}}%
\vskip-9pt
\bgroup\parskip=2pt
{\tocfonts
\hyphenpenalty=10000
    \@starttoc{lot}%
}
\egroup
    \if@restonecol\twocolumn\fi
\newpage
\markboth{}{}
    }



\newcommand{\listoffigures}{%
\startonoddpage
\global\titletrue
    \chapter*{List of Figures
        \@mkboth{\uppercase{List of Figures}}{\uppercase{List of Figures}}}%
\vskip-9pt
\bgroup\parskip=2pt
{\tocfonts
\hyphenpenalty=10000
    \@starttoc{lof}%
}
\egroup
    \if@restonecol\twocolumn\fi

\newpage
\markboth{}{}
    }

\def\l@part#1#2{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \setlength\@tempdima{3em}%
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
         Part #1\hfil\hbox to\@pnumwidth{\hss %#2
}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}
    \endgroup
  \fi}


\def\l@chapter#1#2{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 14pt \@plus\p@
     \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode
      \advance\leftskip\@tempdima
      \hskip -\leftskip
 #1\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}

% \@dottedtocline{LEVEL}{INDENT}{NUMWIDTH}{TITLE}{PAGE} :
%   Macro to produce a table of contents line with the following
%   parameters:
%     LEVEL    : If LEVEL > \c@tocdepth, then no line produced.
%     INDENT   : Total indentation from the left margin.
%     NUMWIDTH : Width of box for number if the TITLE has a
%                \numberline command.
%                As of 25 Jan 88, this is also the amount of extra indentation
%                added to second and later lines of a multiple line entry.
%     TITLE    : Contents of entry.
%     PAGE     : Page number.
%
%  Uses the following parameters, which must be set by the document style.
%  They should be defined with \def's.
%    \@pnumwidth : Width of box in which page number is set.
%    \@tocrmarg  : Right margin indentation for all but last line of
%                  multiple-line entries.
%    \@dotsep    : Separation between dots, in mu units.  Should be \def'd to
%                  a number like 2 or 1.7
%
\def\@dottedtocline#1#2#3#4#5{%
\ifnum #1>\c@tocdepth \else \vskip \z@ plus.2\p@ {\leftskip #2\relax
\rightskip \@tocrmarg  \parfillskip -\rightskip \parindent #2\relax
\@afterindenttrue \interlinepenalty \@M \leavevmode \@tempdima #3\relax
\advance \leftskip \@tempdima \hbox {}\hskip -\leftskip #4%
\nobreak% \leaders \hbox {$\m@th \mkern
% \@dotsep mu.\mkern \@dotsep mu$}
\hfill\nobreak\hbox to\@pnumwidth {\hfil #5}\par }\fi}

\newskip\tabandfigskip
\tabandfigskip=6pt plus 3pt minus 3pt
\def\l@table{\setlength\@tempdima{18pt}%
\vskip\tabandfigskip
\@dottedtocline{1}{1.5em}{2.5em}}
\def\l@figure{\setlength\@tempdima{18pt}%
\vskip\tabandfigskip
\@dottedtocline{1}{1.5em}{2.5em}}

\def\l@section{\setlength\@tempdima{18pt}%
\@dottedtocline{1}{1.5em}{2.5em}}%was 2.3em
\def\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
\def\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\def\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\def\l@subparagraph{\@dottedtocline{5}{12em}{6em}}


%%% <<=== end TOC

%%% Bibliography, References, Acknowledgements ===>>

\def\@biblabel#1{#1.}


%%%%%% Chapter Bibliography Using BibTeX

\def\chapbblname#1{\def\currentfilename{#1}}

\def\chapbibliography#1{%
\if@filesw \immediate \write \@auxout {\string \bibdata {#1}}\fi
\expandafter\ifx\csname currentfilename\endcsname\relax
\typeout{^^J^^J As soon as your .bbl file is written,
^^J^^J
!! Please write \string\chapbblname\string{<Your .bbl File Name>\string} !!
^^J\space\space\space before \string\chapbibliography{#1}
^^J^^J
The BibTeX Chapter Bibliography, \string\chapbibliography^^J
\space\space\space Will not print unless you supply this file name!
^^J^^J
}
\else
{\let\thebibliography\chapreferences
\let\endthebibliography\endchapreferences
\@input {\currentfilename.bbl}}\fi
}

\def\thechapbibliography{\section*{References}
\gdef\rrunninghead{\uppercase{REFERENCES}}
 \addcontentsline{toc}{section}{\protect 
 \numberline{References\string\hss}}
\everypar={}
\bgroup
\frenchspacing
\normalsize
\parindent=-1pc
\leftskip=1pc
\vskip1sp
}

\def\endthechapbibliography{\vskip24pt\egroup\parindent=0pt\relax}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\newblock{}



\def\chapreferences{\ifvmode\vskip-\lastskip\fi
\vskip8pt\vskip1sp\section*{REFERENCES}
\addcontentsline{toc}{section}{\protect \numberline{}}
\bgroup
\leftmargini=17pt
\labelsep=3pt
      \list{\@biblabel{\arabic{enumiv}}}%
           {\settowidth\labelwidth{\@biblabel{10.}}}%
\usecounter{enumiv}
}

\expandafter\def\csname chapreferences*\endcsname{\ifvmode\vskip-\lastskip\fi
\vskip8pt\vskip1sp\section*{REFERENCES}
\bgroup
\leftmargini=17pt
\labelsep=3pt
      \list{\@biblabel{\arabic{enumiv}}}%
           {\settowidth\labelwidth{\@biblabel{10.}}}%
\usecounter{enumiv}
}

\def\endchapreferences{\endlist\egroup}
\expandafter\def\csname endchapreferences*\endcsname{\endlist\egroup}

\def\namedchapreferences#1{\ifvmode\vskip-\lastskip\fi
\vskip8pt\vskip1sp\section*{REFERENCES}
\addcontentsline{toc}{chapter}{\protect \numberline{References\string\hss}}
\bgroup
\def\@biblabel##1{[##1]}
\list
  {\@biblabel{\arabic{enumiv}}}{\settowidth\labelwidth{\@biblabel{[#1]}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \def\theenumiv{\arabic{enumiv}}}%
    \def\newblock{\hskip .11em plus.33em minus.07em}%
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=\@m}

\def\endnamedchapreferences{\endlist\egroup}

\def\references{
\startonoddpage
\global\titletrue
\chapter*{References}
\markboth{REFERENCES}{REFERENCES}
\addcontentsline{toc}{chapter}{\protect \numberline{References\string\hss}}
\bgroup
\leftmargini=17pt
\labelsep=3pt
      \list{\@biblabel{\arabic{enumiv}}}%
           {\settowidth\labelwidth{\@biblabel{10.}}}%
\usecounter{enumiv}
}

\def\endreferences{\endlist\egroup}


\def\thebibliography#1{\startonoddpage
\global\titletrue
\chapter*{References}
\markboth{REFERENCES}{REFERENCES}
\addcontentsline{toc}{chapter}{\protect \numberline{\string\hss}References}%% correct this command for Reference hyperlink
\list
  {\@biblabel{\arabic{enumiv}}}{\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \def\theenumiv{\arabic{enumiv}}}%
    \def\newblock{\hskip .11em plus.33em minus.07em}%
    \sloppy\clubpenalty4000\widowpenalty4000
    \sfcode`\.=\@m}

\def\endthebibliography{%
  \endlist}


\def\namedreferences#1{\thebibliography{[#1]}
\def\@biblabel##1{[##1]}
}
\def\endnamedreferences{\endthebibliography}

%% end Bibliography, References and Acknowledgments


%%% Footnotes ===>>>

%  \renewcommand{\footnoterule}{%
%  \kern-3\p@
%  \hrule width .4\columnwidth
%  \kern 2.6\p@}

  \let\footnoterule\relax

\@addtoreset{footnote}{chapter}

\long\def\@makefntext#1{%
    \parindent 1em%
    \noindent
    \@makefnmark#1}

\newcount\footnum
\newcount\savefootnum
\def\endnote#1{\global\advance\footnum by 1\relax$^{\the\footnum}$%
\long\expandafter\gdef\csname foot\the\footnum\endcsname{%
\vtop{\footnotesize\hsize=\textwidth\leftskip=12pt\relax\parindent=-12pt
\indent\hbox to12pt{\the\footnum.\hfill}{#1\strut\vskip1sp}}\vskip1pt}}

\def\endnotetext#1{\global\advance\footnum by 1\relax
\long\expandafter\gdef\csname foot\the\footnum\endcsname{%
\vtop{\footnotesize\hsize=\textwidth\leftskip=12pt\relax\parindent=-12pt
\indent\hbox to12pt{\the\footnum.\hfill}{#1\strut\vskip1sp}}\vskip1pt}}

\newif\ifrefnoteused
\def\refnote{\ifrefnoteused\else\global\refnoteusedtrue
\typeout{^^J^^J!!! ===>>>^^J
Please do not use the command \string\refnote.^^J
Instead write a standard LaTeX reference and use \string\cite
^^J<<<=== !!!^^J^^J}\fi}

\def\notes{\ifnum\footnum=0\else
\savefootnum=\footnum
\footnum=0
\noindent\loop\ifnum\savefootnum>\footnum%
\global\advance\footnum by1\relax%
\csname foot\the\footnum\endcsname\relax%
\expandafter\gdef\csname foot\the\footnum\endcsname{\relax}\relax%
\repeat\global\footnum=0\relax\fi}

\def\note{\hfill}


%%% <<<=== End Footnotes



%%% Running heads ===>>>

%% unless we need these, leave these uncommented
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble

\def\ps@headings{\let\@mkboth\markboth
 \def\@oddfoot{\iftitle\global\titlefalse{\hfill\foliofont\thepage}\fi}
 \def\@evenfoot{\iftitle\global\titlefalse{\foliofont\thepage\hfill}\fi}
 \def\@evenhead{\iftitle\else{\foliofont \thepage}\hskip1.5pc%
\runningheadfont\leftmark\fi\hfill}
 \def\@oddhead{\hfill\iftitle\else{\runningheadfont \rightmark}\hskip1.5pc%
\foliofont\thepage\fi}%
 \def\chaptermark##1{\markboth {\uppercase{##1}}{}}%
 \def\sectionmark##1{\markright {\uppercase{##1}}}}

%%%% uncomment lines in the range of [1476:1484]  and comment [1487-1496]
%%%% to restore the default running head - Anh Huy Phan - 7/01/2009
%%
%%\def\ps@headings{\let\@mkboth\markboth
%% \def\@oddfoot{\iftitle\global\titlefalse{\hfill\foliofont\thepage}\fi}
%% \def\@evenfoot{\iftitle\global\titlefalse{\foliofont\thepage\hfill}\fi}
%% \def\@evenhead{\iftitle\else{\foliofont \thepage}\hskip1.5pc%
%%\runningheadfont\leftmark\fi\hfill}
%% \def\@oddhead{\hfill\iftitle\else{\runningheadfont \rightmark}\hskip1.5pc%
%%\foliofont\thepage\fi}%
%% \def\chaptermark##1{\markboth {\uppercase{Cichocki} and \uppercase{Zdunek} and \uppercase{Phan} and \uppercase{Amari}}{}}%
%% \def\sectionmark##1{\markright {\uppercase{##1}}}}

\markboth{}{}


\ps@headings

\pagestyle{headings}
\pagenumbering{roman}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Special Use Fonts
%%

%% Computer Modern font calls. These will be redefined
%% if PostScript fonts are used.
%%
%% \documentstyle[w-bookps]{w-book} will use PostScript fonts.
%% \documentstyle{w-book} will use Computer Modern fonts.


%% Times-Roman
\def\timesroman{cmr10 }
\def\timesbold{cmbx10 }
\def\timesitalic{cmti10 }
\def\timesbolditalic{cmbx10 }

%% Helvetica
\def\helvetica{cmss10 }
\def\helveticabold{cmssbx10 }
\def\helveticaboldoblique{cmssi10 }
\def\helveticaoblique{cmssi10 }

%% Table of Contents
\font\tocarttitlefont=\timesitalic at 12pt
\font\tocfonts=\timesitalic at 12pt

%% Running heads
\font\runningheadfont=\helveticaoblique at8pt
\font\foliofont=\helveticaboldoblique at 9pt

%% Book Title, Half title page

\font\halftitlefont=\timesitalic at 18pt
\font\titlefont=\timesitalic at 30pt
\font\subtitlefont=\timesitalic at 20pt
\font\editionfont=\helvetica at 14pt
\font\authorfont=\timesbold at 12pt

\font\wileyfont=\timesbolditalic at 14pt
\font\imprintfont=\helvetica at 9pt

%% Part Title
\font\partfont=\timesitalic at 30pt
\font\partnumberfont=\timesitalic at 20pt

%% Chapter Title
\font\chapternumberfont=\timesitalic at 60pt
\font\chaptertitlefont=\timesitalic at 30pt

%% Section heads
\font\sectionfont=\helveticabold at 10pt
\font\subsectionfont=\helveticabold at 10pt
\font\subsubsectionfont=\helveticaboldoblique at 10pt
\font\paragraphfont=\helveticaoblique at 10pt

%% Caption
\font\fignumfont=\helveticaboldoblique at 9pt
\font\tabnumfont=\helveticaboldoblique at 9pt
\font\figtextfont=\timesroman at 9pt
\font\tabtextfont=\timesroman at 9pt

%% Table fonts
\def\tablefontsize{\footnotesize}
\def\tablefootnotesize{\scriptsize}

%% References/Bibliography
\def\bibfontsize{\small}

%% Text fonts
\font\bit=\timesbolditalic at 8pt
\font\scripttt=cmtt10 scaled 700

%% Fake Small Caps
\font\bigscfont\timesroman at 9.5pt
\font\smscfont\timesroman at 7.5pt
\font\wherefont\timesitalic at 8pt

%% Problem Set
\font\probfont\timesbolditalic at 10pt

%% Color font
\usepackage[dvipsnames,usenames]{color}
%%\definecolor{sectioncolor}{rgb}{0.2,0.7,0.2}
%%\definecolor{paragraphcolor}{rgb}{0.2,0.2,0.7}
%%\definecolor{subsectioncolor}{rgb}{0.2,0.3,0.6}
%%\definecolor{subsubsectioncolor}{rgb}{0.5843   , 0.3882   , 0.3882}
%%\definecolor{tablecolor}{rgb}{0.2,0.7,0.2}
%%\definecolor{figurecolor}{rgb}{0.2,0.7,0.2}
%%\definecolor{redaccent}{rgb}{0.3882  ,  0.1412  ,  0.1373}
%%\definecolor{aqua}{rgb}{0 , 0.4118, 0.5686}%%{0.09,0.211,0.37}
%%\definecolor{textcolor}{rgb}{0 , 0.4118, 0.5686}
%%\definecolor{captioncolor}{rgb}{0 , 0.4118, 0.5686}


\definecolor{aqua}{rgb}{0 , 0.4118, 0.5686}%%{0.09,0.211,0.37}
\definecolor{violet}{rgb}{0.4706 ,   0.3137 ,   0.6667}
\definecolor{violetdark}{rgb}{0.3686  ,  0.2353   , 0.5216}
\definecolor{milkblue}{rgb}{0.2,0.2,0.7}
\definecolor{redaccent}{rgb}{0.3882  ,  0.1412  ,  0.1373}
\definecolor{bluedark}{rgb}{0.2,0.3,0.6}
\definecolor{QuartzPurple}{rgb}{0.302,	0.235,	0.573}
\definecolor{SapphireBlue}{rgb}{0.000,	0.388,	0.671}



\newcommand\txtcolor            {\color{QuartzPurple}}
\newcommand\captioncolor        {\color{QuartzPurple}}
\newcommand\tablecolor          {\color{aqua}}
\newcommand\figurecolor         {\color{aqua}}
\newcommand\chaptercolor        {\color{SapphireBlue}}
\newcommand\sectioncolor        {\color{SapphireBlue}}
\newcommand\paragraphcolor      {\color{SapphireBlue}}
\newcommand\subsectioncolor     {\color{SapphireBlue}}
\newcommand\subsubsectioncolor  {\color{SapphireBlue}}
%\newcommand\toccolor            {\color{QuartzPurple}}



%% setting color for normal text
\txtcolor
\renewcommand\normalcolor{\txtcolor}

%%% <=== end special use fonts

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Fake Small Caps

% To use: \fakesc some stuff*
%% \bigscfont, \smscfont are listed in font section above.

\def\fakesc#1*{\edef\stuff{#1 }%
\expandafter\pickupnewword\stuff* }
\def\saveast{*}

\def\rrelax{\relax}

\def\pickupnewlett#1{\def\letter{#1}%
\ifx\letter\saveast\unskip%
\let\go\relax \else%
\ifx\letter\rrelax{\color{red}\bigscfont\ }%
\let\go\pickupnewword\else%
\uppercase{\def\ucletter{#1}%  upper case letter
\ifx\letter\ucletter%
{\bigscfont#1}\else{\smscfont#1}\fi}%%
\let\go\pickupnewlett\fi\fi\go}

\def\pickupnewword#1 {\expandafter\pickupnewlett#1\relax}
%% <== end fakesc

%% Algorithm

\def\algorithmcode{\bgroup\obeylines\obeyspaces\def\ {\quad}
\footnotesize\tt\leftskip=2pc\vskip4pt\relax}
\def\endalgorithmcode{\vskip4pt\egroup}

%% <== end algorithm


%% ==> glossary
\def\glossary{\startonoddpage\global\titletrue
\markboth{GLOSSARY}{GLOSSARY}
\chapter*{Glossary}
\addcontentsline{toc}{chapter}{\protect \numberline{Glossary\string\hss}}
\everypar={}
\bgroup\leftskip=1pc \parindent=-1pc}
\def\term#1{\vskip2pt{\bf #1}\hskip1em\relax}
\def\endglossary{\vskip12pt\egroup}
%% <==


%%%%%%%%%%%%%%%%%%%%%%%
%% Index ===>>>

% latex default

\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\uppercase{\indexname}}%
                        {\uppercase{\indexname}}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand{\@idxitem}  {\par\hangindent 40\p@}
\newcommand{\subitem}   {\par\hangindent 40\p@ \hspace*{20\p@}}
\newcommand{\subsubitem}{\par\hangindent 40\p@ \hspace*{30\p@}}
\newcommand{\indexspace}{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
%%% <<<=== end latex default Index

%%%%%%% Index Formatting

\def\testforadd{\addtoindex}

\def\seealso#1{\hskip16pt{\it See also\ }{#1}}

\def\ltr#1{\nobreak\vskip5pt
\hrule
\nobreak
\vskip3pt
{\bf#1}
\nobreak
\vskip5pt}

\let\saveangle>

\newif\ifnosrtfile
\def\newinput#1{\openin1 #1 \ifeof1 \typeout{^^J^^JWarning:
There is no file #1.^^J^^J\space\space \string\printindex\space
will not work without a \jobname.srt file.^^J^^J
\space\space Please run your sorting program
on \jobname.inx\space
to produce \jobname.srt.^^J^^J}\global\nosrtfiletrue\else
\global\nosrtfilefalse\closein1 \relax\input #1\fi}

{\catcode`>=\active
\gdef\printindex{
\catcode`>=\active%
\let>\doindex
\newinput{\jobname.srt} \catcode`\*=12 \catcode`>=12
\ifnosrtfile\let\doitnow\relax\else\let\doitnow\doindexending\fi
\doitnow
}

\gdef\doindexending{
\vskip-14pt
\endmulticols
\newpage
\onecolumn
\thispagestyle{empty}
\pagestyle{empty}
\leftskip=0pt\vskip1sp}

}
\newif\if@restonecol
\let\saveanglebracket>

{\catcode`>=\active%
\newif\if@restonecol
\gdef\doindex #1{
\clearpage
\multicols{2}[\chapter*{Index}]
\markboth{INDEX}{INDEX}
\bgroup
\let>\saveanglebracket
\footnotesize
\noindent\ \unskip
\vskip-\baselineskip
\parindent=-\indexwrap \leftskip=\indexwrap
\def\,{\ifmmode\typeout{mmode^^J^^J}\mskip\thinmuskip\else\comma\fi}
 \addcontentsline{toc}{chapter}{\protect
 Index\numberline{\string\hss}} %% change here for Index hyperlink
\columnsep 35pt
\hfuzz=.5in
\def\checker{#1}
\widowpenalty0
\clubpenalty0
\parskip=0pt plus.001pt
\raggedright
\ifx\checker\testforadd\let\next=\relax\else\let\next=\beginindex\fi\next #1}
}



%%%%%%  Index Entries

\newwrite\innx

\immediate\openout\innx=\jobname.inx

\newif\ifsilent
\newif\ifnotsilent

\newcount\spcount
\def\oneast{\global\spcount=1}
\def\twoast{\global\spcount=2}

\def\astsplit#1*#2#3{{\xdef\pagenumzeros{\ifnum\count0<100
\ifnum\count0<10 00\else 0\fi\fi}\if#2\relax%
\def\,{\noexpand\comma}%
\writeit{#1}%
\else\if#3\relax
\def\,{\noexpand\comma}%
\writeita{#1}\else
\def\,{\noexpand\comma}%
\writeitb{#1}\fi\fi%
\ifnotsilent#1\global\notsilentfalse\fi}}

\def\pickup#1{\astsplit#1*\relax\relax}
\def\inx#1{\global\notsilenttrue\pickup{#1}}
\def\inxx#1{\astsplit#1*\relax\relax}

\def\romannums{\csname @roman\endcsname\c@page}

\def\writeit#1{\xdef\pagenumzeros{\ifnum\count0<100
\ifnum\count0<10 00\else 0\fi\fi}%
\ifx\thepage\romannums%
\write\innx{\noexpand#1 |{%
\pagenumzeros\the\count0\string\global\string\romanontrue}}\else
\write\innx{\noexpand#1 |{\pagenumzeros\the\count0}}\fi}

\def\writeita#1{\xdef\pagenumzeros{\ifnum\count0<100
\ifnum\count0<10 00\else 0\fi\fi}%
\ifx\thepage\romannums%
\write\innx{\noexpand#1 |{\pagenumzeros\the\count0
\oneast\string\global\string\romanontrue}}\else
\write\innx{\noexpand#1 |{\pagenumzeros\the\count0
\oneast}}\fi}

\def\writeitb#1{\xdef\pagenumzeros{\ifnum\count0<100
\ifnum\count0<10 00\else 0\fi\fi}%
\ifx\thepage\romannums%
\write\innx{\noexpand#1 |{\pagenumzeros\the\count0
\twoast\string\global\string\romanontrue}}\else
\write\innx{\noexpand#1 |{\pagenumzeros\the\count0
\twoast}}\fi}

\write\innx{\noexpand>}
\write\innx{\noexpand~}

%% end sending to index

%% start formatting index

%%% Index Macros
%%% Copyright 1992, Amy Hendrickson
%%% TeXnology, Inc.
%%% All rights reserved.


\def\indexit#1{{\it\expandafter\capthis #1}}
\def\indexbf#1{{\bf\expandafter\capthis #1}}
\def\indexsl#1{{\sl\expandafter\capthis #1}}
\def\indextt#1{{\tt\expandafter\capthis #1}}

\def\lcindexit#1{{\it #1}}
\def\lcindexbf#1{{\bf #1}}
\def\lcindexsl#1{{\sl #1}}
\def\lcindextt#1{{\tt #1}}


\def\module{}\def\rightmodule{}
\def\splittocstuff#1{\global\indxnum=#1}

\def\onward{}

\def\addtoindex{\par}

\def\endadd #1{\def\lfe{#1}\if\lfe\noexpand~
\let\more=\relax\let\lfe=\endindx\else
\ifx\lfe\munge\let\more=\relax\else
\ifx\lfe\testforadd\vskip1pt\let\more\relax\else
\par\let\more=\beginindex\fi\fi\fi
\expandafter\more\lfe}

\newcount\indxnum
\newcount\oldindxnum

\def\comma{,\ }
\def\go{}
\def\compare{}
\def\eatthree#1,#2,{}
\def\eattwo#1,{}
\def\oldthreecol{}
\def\oldtwocol{}
\def\oldonecol{}

\newbox\savenumbox

\def\eatcommas{\ifx\twocol\empty%
\let\eat=\relax
\else
   \ifx\threecol\empty\let\eat=\eattwo\else
      \let\eat=\eatthree
    \fi
\fi\eat}

\def\splitoff#1,#2,#3,{\gdef\onecol{#1}\gdef\twocol{#2}%
\gdef\threecol{#3}\eatcommas}
\def\endin{}
\def\testind{\endin}
\def\testadd{\addtoindex}
\def\oldcompare{}
\newif\ifendindex
\newif\ifsavenum
\newcount\saveindxnum
\newif\ifnnum
\newdimen\indexwrap
\newdimen\indexindent


\def\endindx{\vskip1sp\egroup
\@normalsize}

\newif\ifromanon

%+++ Important changes here, made 9/24/99
\def\beginindex #1 |#2 #3{%
\let~\relax
\parindent=-\indexwrap \leftskip=\indexwrap%
\def\module{}\let\oldrightmodule=\rightmodule%
\def\rightmodule{}%
\setbox0=\hbox{\expandafter\splittocstuff{#2}}%
\global\oldindxnum=\saveindxnum\relax%
\def\changenums{\ifnum\spcount<1%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
, {\module\ifromanon\expandafter\romannumeral\fi%
\number\indxnum\rightmodule}\else%
\ifnum\spcount=1%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
, \underbar{\module\ifromanon\expandafter\romannumeral\fi%
\number\indxnum\rightmodule}%
\relax\global\spcount=0%
\else\ifnum\spcount=2%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\relax%
\global\spcount=0\fi%
\global\spcount=0, {\bf\module\ifromanon\expandafter\romannumeral\fi%
\number\indxnum\rightmodule}%
\fi\fi\fi}%
\def\comparenums{\ifnum\oldindxnum=0\changenums\else%%%%
\ifnum\indxnum=\oldindxnum%
\ifx\oldrightmodule\rightmodule\relax%
\else%
, {\module\ifromanon\expandafter\romannumeral\fi%
\number\indxnum\rightmodule}\fi%
\else%
\advance\oldindxnum by 1\relax%
\ifnum\indxnum=\oldindxnum%
\ifnum\spcount=1%
\gdef\numinbox{\underbar{\ifromanon\expandafter\romannumeral\fi%
\number\indxnum\rightmodule}}%
\global\nnumtrue%
\else%
\ifnum\spcount=2\gdef\numinbox{\bf\ifromanon\expandafter\romannumeral\fi%
\number\indxnum\rightmodule}%
\global\nnumtrue%
\else%
\gdef\numinbox{\ifromanon\expandafter\romannumeral\fi%
\number\indxnum\rightmodule}\fi\fi%
\setbox\savenumbox=\hbox{--\hskip1sp\numinbox}%
\global\spcount=0%
\global\savenumtrue%
    \else%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\changenums%
\else%
\changenums\fi\fi\fi\fi}%
\def\compare{#1}% now compare words
\ifx\compare\oldcompare\comparenums%
\else\splitoff#1,{},{},%
%%
\ifx\onecol\oldonecol%
   \ifx\twocol\empty%
       \ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
   \par\global\oldindxnum=0
   \expandafter\capthis\onecol\comparenums%
\else%%
         \ifx\twocol\oldtwocol%
      \ifx\threecol\empty\else%
       \ifx\threecol\oldthreecol
                    \else%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
\ifx\onecol\oldonecol\else
\par\expandafter\capthis%
\onecol\fi%
\ifx\twocol\oldtwocol\else\par\hskip\indexindent\twocol\fi%
\par\global\oldindxnum=0\relax%
\hskip2\indexindent\threecol\comparenums%% ***
\fi\fi\else%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
\global\oldindxnum=0\relax
\ifx\threecol\empty%
%%
\ifx\onecol\oldonecol\else\par\expandafter\capthis\onecol\fi%
\par\hskip\indexindent\twocol\comparenums%% ***
\else%% this is just for error control,
%% when author has new third level
%% but neglects second level entry.
\ifx\onecol\oldonecol\else\par%
\expandafter\capthis\onecol\fi%
\ifx\twocol\oldtwocol\else
\par\hskip\indexindent\twocol\fi%
\par\hskip2\indexindent\threecol\comparenums% ***
\fi%
\fi\fi\else\ifx\twocol\empty%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
\par\global\oldindxnum=0\relax%
\expandafter\capthis\onecol\comparenums%
\else\ifx\threecol\empty%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
\global\oldindxnum=0\relax%
\ifx\onecol\oldonecol\else\par%
\expandafter\capthis\onecol\fi%
\par\hskip\indexindent\twocol\comparenums%% ***
\else%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
\global\oldindxnum=0\relax%
\ifx\onecol\oldonecol\else\par%
\expandafter\capthis\onecol\fi%
\ifx\twocol\oldtwocol\else
\par\hskip\indexindent\twocol%
\fi%
\par\hskip2\indexindent\threecol\comparenums% ***
\fi\fi\fi\fi%
\gdef\oldcompare{#1}%
\global\saveindxnum=\indxnum%
\ifnnum\global\saveindxnum=0\global\nnumfalse\fi%
\global\let\oldonecol=\onecol\relax%
\global\let\oldtwocol=\twocol%
\global\let\oldthreecol=\threecol%
\def\turnoff{#3}\ifx\turnoff\testadd%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi%
\let\go=\relax\else\ifx\turnoff\testind\let\go=\relax%
\ifsavenum\global\savenumfalse\hbox{\unhbox\savenumbox}\fi\else%
\if\turnoff\noexpand~\let\go=\endindx%
\ifsavenum\global\savenumfalse\unskip\hbox{\unhbox\savenumbox}\fi%
\else\let\go=\beginindex\fi\fi\fi\global\romanonfalse\go#3}



\let\lastonecol\relax
\let\lasttwocol\relax
\let\lastthreecol\relax
\let\oldonecol\relax
\let\oldtwocol\relax
\let\oldthreecol\relax


%% CHANGE INDEX FORMAT HERE ====>

%% If you DO NOT want the first letter of each first level index
%% entry to be capitalized, delete the % in front of the following line,
%% and put % in front of the next line:

%\let\capthis\relax
\def\capthis#1{\uppercase{#1}}%

\indexindent=8pt %% indentation for index subentries
\indexwrap=24pt  %% indentation when term is too wide for column,
                 %% continues on following lines indented this much.

%%%%%
%% Frank Mittlebach's multicol macros, minus comments and
%% irrelevancies.

\def\multicols#1{\col@number#1\relax
  \@ifnextchar[\mult@cols{\mult@cols[]}}

\def\mult@cols[#1]{\@ifnextchar[%
  {\mult@@cols{#1}}%
  {\mult@@cols{#1}[\premulticols]}}

\def\mult@@cols#1[#2]{%
   \enough@room#2%
   #1\nobreak\par%\addvspace\multicolsep
   \begingroup
   \prepare@multicols\ignorespaces}

\def\enough@room#1{\par \penalty\z@
   \page@free \pagegoal
   \advance \page@free -\pagetotal
   \ifdim \page@free <#1\newpage \fi}

\def\prepare@multicols{%
  \output{\global\setbox\partial@page
                 \vbox{\unvbox\@cclv}}\eject
  \vbadness9999 \hbadness5000
  \tolerance\multicoltolerance
  \doublecol@number\col@number
  \multiply\doublecol@number\tw@
  \advance\baselineskip\multicolbaselineskip
  \advance\@colroom-\ht\partial@page
  \vsize\col@number\@colroom
  \advance\vsize\c@collectmore\baselineskip
  \hsize\columnwidth \advance\hsize\columnsep
  \advance\hsize-\col@number\columnsep
  \divide\hsize\col@number
  \linewidth\hsize
  \output{\multi@columnout}%
  \multiply\count\footins\col@number
  \multiply\skip \footins\col@number
  \reinsert@footnotes}

\def\endmulticols{\parskip=0pt
\par\penalty\z@
  \output{\balance@columns}\eject\nobreak%
\endgroup
}

\newcount\c@unbalance     \c@unbalance   = 0
\newcount\c@collectmore   \c@collectmore = 0

\newcount\col@number
\newcount\doublecol@number
\newcount\multicoltolerance
                   \multicoltolerance = 9999
\newdimen\page@free
\newdimen\premulticols  \premulticols = 50pt
\newdimen\postmulticols \postmulticols= 50pt
\newskip\multicolsep
      \multicolsep =0pt

\newskip\multicolbaselineskip
                   \multicolbaselineskip=0pt
\newbox\partial@page

\def\process@cols#1#2{\count@#1\relax
     \loop #2%
     \advance\count@\tw@
     \ifnum\count@<\doublecol@number
   \repeat}


\def\page@sofar{\unvbox\partial@page
   \process@cols\z@{\wd\count@\hsize}%
\hbox to\textwidth{%
     \process@cols\tw@{\box\count@\hss}%
\box\z@}
}


\def\reinsert@footnotes{\ifvoid\footins\else
         \insert\footins{\unvbox\footins}\fi}

\def\multi@columnout{%
   \ifnum\outputpenalty <-\@Mi
   \speci@ls \else
   \splittopskip\topskip
   \splitmaxdepth\maxdepth
   \dimen@\@colroom
   \divide\skip\footins\col@number
   \ifvoid\footins \else
      \advance\dimen@-\skip\footins
      \advance\dimen@-\ht\footins   \fi
   \process@cols\tw@{\setbox\count@
            \vsplit\@cclv to\dimen@}%
   \setbox\z@\vsplit\@cclv to\dimen@
   \ifvoid\@cclv \else
       \unvbox\@cclv
       \penalty\outputpenalty\fi
   \setbox\@cclv\vbox{\page@sofar}%
   \@makecol\@outputpage
   \global\@colroom\@colht
   \process@deferreds
   \global\vsize\col@number\@colroom
   \global\advance\vsize
      \c@collectmore\baselineskip
   \multiply\skip\footins\col@number\fi}

\def\speci@ls{%
   \unvbox\@cclv\reinsert@footnotes
   \gdef\@currlist{}}

\def\process@deferreds{%
   \@floatplacement
   \begingroup
    \let\@tempb\@deferlist
    \gdef\@deferlist{}%
    \let\@elt\@scolelt
      \@tempb \endgroup}

\newif\ifshr@nking

\def\raggedcolumns{%
   \@bsphack\shr@nkingtrue\@esphack}

\def\flushcolumns{%
   \@bsphack\shr@nkingfalse\@esphack}

\newcount\loopcount

\newdimen\savedimen
\newdimen\fixdimen

\def\escapeloop{\gdef\iterate{}}

\let\saveiterate\iterate

\def\balance@columns{%
\loopcount=0
\fixdimen=0pt
   \splittopskip\topskip
   \splitmaxdepth\maxdepth
   \setbox\z@\vbox{\unvbox\@cclv}\dimen@\ht\z@
   \advance\dimen@\col@number\topskip
   \advance\dimen@-\col@number\baselineskip
   \divide\dimen@\col@number
   \advance\dimen@\c@unbalance\baselineskip
   {\vbadness\@M \loop
   {\process@cols\@ne{\global\setbox\count@
                              \box\voidb@x}}%
     \global\setbox\@ne\copy\z@
     {\process@cols\thr@@{\savedimen\ht\@ne \advance\savedimen by-\dimen@
\global\setbox\count@ \vsplit\@ne to\dimen@
\ifdim\savedimen>\ht\@ne % AH: TeX was forced to make box bigger than \dimen@.
\advance\savedimen by-\ht\@ne \advance\savedimen by -7.1pt
\ifdim\savedimen>\fixdimen \global\fixdimen\savedimen\fi\fi}}%
    \ifdim\ht\@ne >\ht\thr@@
    \global\advance\dimen@\p@%.5\baselineskip
\global\advance\loopcount by1
\ifnum\loopcount=90 \escapeloop\fi
   \repeat}%
\let\iterate\saveiterate
\dimen@\ht\thr@@
\ifdim\fixdimen>0pt \advance\dimen@ by \fixdimen\fi
   \process@cols\z@{\@tempcnta\count@
        \advance\@tempcnta\@ne
        \setbox\count@\vtop to\dimen@
           {\unvbox\@tempcnta
           \ifshr@nking\vfill\fi}
}%
   \global\vsize\@colroom
   \global\advance\vsize\ht\partial@page
   \page@sofar
}

%%%%%%%%%%%%%%
% Draft

\def\draft{\global\drafttrue
\def\@oddfoot{\vtop to 0pt{\vskip24pt\large\tt
D R A F T\hfill\today, \realtime\hfill D R A F T\global\titlefalse\vss}}
\def\@evenfoot{\@oddfoot}\normalsize}

\def\fix{\ifcase\oldtime 0\or0\or0\or%
0\or0\or0\or0\or0\or0\or0\fi}
\def\fixtiming{\ifcase\timing 0\or0\or0\or%
0\or0\or0\or0\or0\or0\or0\fi}
\newcount\timing
\newcount\hourcount
\newcount\oldtime

\def\realtime{\timing=\time \oldtime=\time
\ifnum\timing>60 \divide\timing by 60
\hourcount=\the\timing
\multiply\timing by 60
\advance\oldtime by-\timing
\ifnum\hourcount<12 \number\hourcount:\fix\number\oldtime am\fi%
\ifnum\hourcount=12 \number\hourcount:\fix\number\oldtime pm\fi%
\ifnum\hourcount>12 \advance\hourcount by-12
\number\hourcount:\fix\number\oldtime pm\fi
\else12:\fixtiming\number\timing am\fi}

\newcommand{\today}{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}

%%% Two column, landscape, environment for program examples

%% To use:
% \programcaption{<Caption>} % (optional)
% \colone
% \begin{verbatim}
%  (program code)
% \end{verbatim}
% \endcolone
% \coltwo
% \begin{verbatim}
% (program code)
% \end{verbatim}
% \endcoltwo

\newdimen\landscapeoddpagedown
\newdimen\landscapeevenpagedown
\newdimen\landscapepagetoright

%% These dimensions are used for positioning the landscape page
%% on the physical page.
\landscapeoddpagedown=10pt %% move code down on odd numbered page
\landscapeevenpagedown=79pt%% move code down on even numbered page
\landscapepagetoright=1in  %% move code to the right on either odd or
                           %% even pages

\def\programcaption#1{\def\thecaption{#1}}
\programcaption{}
\def\doprogramcaption{\ifx\thecaption\empty
\else
\def\@captype{figure}
\caption{\thecaption}\fi
\gdef\thecaption{}}
\def\colone{\vglue36pt\bgroup\textwidth 47pc
\oddsidemargin=0pt
\evensidemargin=0pt
\thispagestyle{empty}
\setbox1=\vbox\bgroup
\xscriptsize\let\tt\scripttt
\hsize=22.5pc\vskip-6pt\vskip1sp}
\def\endcolone{\vskip-6pt\egroup}
\def\coltwo{\setbox2=\vbox\bgroup\xscriptsize\let\tt\scripttt\hsize=22.5pc\vskip-6pt
\vskip1sp}

\def\endcoltwo{\vskip-12pt\egroup
\ifodd\c@page \global\dimen0=\landscapeoddpagedown
\else\global\dimen0=\landscapeevenpagedown\fi
\vglue\dimen0
\moveright-\landscapepagetoright\vbox{%% for whole page
\hrule height .9pt
\hbox{%%
\vrule height 27pc width .9pt
\vbox to27pc{\hbox to23.5pc{\hss\hskip6pt\vbox{\unvbox1}\hss}
\vfill}
\vrule
\vbox to27pc{\hbox to23.5pc{\hss\hskip6pt\vbox{\unvbox2}\hss}
\vfill}\vrule height 27pc width .9pt\hss}
\hrule
\vtop to0pt{\hsize=47pc\doprogramcaption\vss}}
\vskip1sp\newpage\egroup}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Problem Set Macros

\newcount\probcount
\newcount\subprobcount

\def\problems{\vskip12pt
\markright{PROBLEMS}
\addcontentsline{toc}{section}{\protect\numberline{}Problems}
\noindent{\probfont Problems}\markright{PROBLEMS}\vskip6pt}

\def\endproblems{\xdef\@currentlabel{\thechapter}}

\def\prob{\vskip6pt
\global\subprobcount=0
\global\advance\probcount by1
\xdef\@currentlabel{\the\c@chapter.\the\probcount}
\noindent{\bf\the\c@chapter.\the\probcount\hskip12pt\relax}}

\def\subprob{\vskip1pt
\global\advance\subprobcount by1
\xdef\@currentlabel{\the\c@chapter.\the\probcount.\the\subprobcount}
{\bf\the\c@chapter.\the\probcount.\the\subprobcount.\hskip6pt\relax}}


%%%% <<==  end of problem set macros


\pagenumbering{roman}
\widowpenalty10000
\clubpenalty10000

%%%%%
%%%% side by side
%%%% to use:
% \begin{figure or table} \sidebyside{}{} \end{figure or table}
%
% \begin{table}
% \sidebyside{\caption{table caption} first table}
% {\caption{table caption} second table}
% \end{table}

% or

% \begin{figure}
% \sidebyside{\vskip<dimen>\caption{fig caption}}
% {\vskip<dimen>\caption{fig caption}}
% \end{figure}

%% To label for cross referencing, use \label after and <outside> of
%% \caption{}

% \begin{table}
% \sidebyside{\caption{table caption}\label{tab1}
% first table}
% {\caption{table caption}\label{tab2} second table}
% \end{table}

% or

% \begin{figure}
% \sidebyside{\vskip<dimen>\caption{fig caption}\label{fig1}}
% {\vskip<dimen>\caption{fig caption}\label{fig2}}
% \end{figure}



\long\def\sidebyside#1#2{%
\hbox to\textwidth{\vtop{\hsize=.5\textwidth%
\advance\hsize by -.5\columnsep
\parindent=0pt
\centering

#1\vskip1sp}\hskip\columnsep\vtop{\hsize=.5\textwidth%
\advance\hsize by -.5\columnsep
\parindent=0pt
\centering
#2

}\hfill}}

\endinput
Oct 18
Changed from \Large to \LARGE in title for math in title
to better match font.

Oct 14
Improvement to \endadd so that it can be
used as the last thing in an index

August 30
Added \markright{PROBLEMS} for problem section.

August 24
More changes to indexing macros, in response to
report of nesting not working correctly.

August 15
Small changes to title and halftitle pages for better
baselineskip

August 14
Yet another try at getting index to work correctly when
author has secondary level of index entry without first
level. Math will still not crash indexing algorithm.

June 20, 1999
Fixed chapnamereferences toc entry so that it matches the other
references in TOC.

May 28, 1999
Got rid of \titletrue in chapter command, until after new page
starts, so that running head will still work on last page of
preceding chapter.

May 26, 1999
Added a little space between entries in List of Figures and List of Tables.

Also made it possible for \begin{theindex}...\end{theindex} to work
for authors that want to use the default LaTeX index commands.

February 15, 1999
Fixed beginindex so that math in index doesn't cause it to
crash, got rid of xdefs.

November 28, 1998
Added List of Tables and List of Figures.

November 24, 1998
Added curlys to definition of astsplit to keep redefinition of
\, from changing thin spaces for rest of document.

November 23
Changes to Appendix so that \\ works; baselineskip is better;
\thanks{} will work in appendix title, but only once-- * is
hard coded in. Should only need it once.

October 9, 1998
Made default setting be \setcounter{tocdepth}{2}
Made \chapter*{} have no running head even if on even numbered page.
Fixed appendix with title in TOC, added a space before title.

September 8, 1998
Many small changes for upgrade

August 31, 1998
Added sidebyside macro;

October 9, 1997
Made extra vertical space at top of table float disappear.
Changed topskip to 10pt, from 0pt

Sept 27, 1997
Added xref capability to \prob and \subprob.

September 20, 1997
debugged how includeonly works in appendix.

September 12, 1997
Added Problem set capability.

August 9, 1997
debugged line break differences when includeonly is used, now
line breaks are the same in either case.

August 8, 1997
added flexible space above and below equations

August 7, 1997
set \@mainmattertrue when chapter is 0, makes chapter toc entry
format correctly; set pagenumbering in appendix to roman for
use in includeonly.

July 25, 1997
Change to \@chapter macro to have \refstepcounter independent
of \@mainmatter being true, to make include and includeonly work
correctly.

Set pagenumbers to arabic with every chapter, set page counter to
1 for chapter 1.

June 30, 1997
Took out \onecolumn which will make first page be too long
if there is nothing but text in it... obscure bug.

May 29, 97
\begin{chapreferences*}...\end{chapreferences*} will make
chapter references that will not send any info to the TOC.

May 22, 97
Footnote change back to 1 at each new page changed.

May 14, 97
Added Acknowledgments and Preface to TOC,
made no hyphenation in TOC, added stretch to tocrmarg
so that long words will go to next line.

March 25--April 8, 1997:
Added \colone,\coltwo for two column algorithm examples

no stretch above and below math, or section heads

equal spacing above and below algorithm

Added fix to index command so that it doesn't give error
messages is there is not .srt file., in w-book and w-edbk.

Increased vertical spacing in multiple line chapter title.
w-book only. (It worked before when \\ was used to start a
new line; now it works the same way if lines are broken
by LaTeX.)

Start chapter and part on new odd page. Chapter
already did that, I added it to part.

Arabic pagenumbering starts with first part, starting with
page 1. First chapter is page 3, unless it is not preceded
with a part in which case it is page 1.

Footnote counter changed so footnotes start at one with
each new chapter.

Appendix running head now will use appendix title, or Appendix A

The cities line is as follows:  New York Chichester  Weinheim
Brisbane Singapore Toronto

Figure and table captions are 1pt larger. Now they are in 9pt.
Math and italics in captions will also be in 9pt.

\namedreferences
Added description of namedreferences and namedchapreferences
in documentation, and example in sample pages.

Appendix xref now working correctly.
