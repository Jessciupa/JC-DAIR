--------------------------------------------------------------------------------
-- Robustness of the dilemma classifier inside a hyperrectangle

-- The classifier maps the 30 reduced embedding dimensions of a Moral Machine
-- dilemma's description to scores for the two choices, stay or swerve. A
-- hyperrectangle encloses the embeddings of a dilemma and its character
-- perturbations. The network is robust on the box if every point inside it gets
-- the choice the cultural cluster made for that dilemma.

--------------------------------------------------------------------------------
-- Inputs

inputSize = 30
type Input = Tensor Real [inputSize]

-- The embeddings were scaled to [0, 1] in every dimension.

validInput : Input -> Bool
validInput x = forall i . 0 <= x ! i <= 1

--------------------------------------------------------------------------------
-- Outputs

-- One score per choice; the choice with the highest score is made.

type Output = Tensor Real [2]
type Label = Index 2

stay   = 0   -- stay on course: protects the characters in the other lane
swerve = 1   -- swerve: protects the characters ahead

--------------------------------------------------------------------------------
-- The network

@network
classifier : Input -> Output

-- The classifier makes choice i for input x.

advises : Input -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- The hyperrectangle

-- One box at a time: a [lower, upper] bound for every input dimension, and the
-- choice a cultural cluster made for the dilemma it was built from. verify.sh
-- passes them in.
-- (Quantifying over all boxes in one property triggers an internal error in
-- Vehicle 0.28, so each box is verified by its own call, as in DAIR-course-NLP.)

@dataset
hyperrectangle : Tensor Real [inputSize, 2]

@parameter
label : Label

inHyperrectangle : Input -> Bool
inHyperrectangle x = forall i . hyperrectangle ! i ! 0 <= x ! i <= hyperrectangle ! i ! 1

--------------------------------------------------------------------------------
-- Property

-- Every point inside the hyperrectangle gets the box's choice.

@property
robust : Bool
robust = forall x . validInput x and inHyperrectangle x => advises x label
