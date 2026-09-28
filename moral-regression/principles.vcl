--------------------------------------------------------------------------------
-- Ethical principle properties of the SaveMe effort network

-- The network is trained on data from `simulation.py`, whose controller is
--
--   effort = kEffort * max(0, score - tau) * max(0, 1 - R / rMax)
--   score  = n_estimated * p_alive * urgency
--
-- The SaveMe principles are NOT part of that controller. Each property below
-- states one principle as a statement about the network's effort, so that
-- verifying it tells us which principles the trained network embodies.

--------------------------------------------------------------------------------
-- Inputs

-- The network takes inputs of the form of a vector of 5 numbers.

type InputVector = Tensor Real [5]

-- Meaningful names for the indices, in the column order of the data.

risk       = 0   -- R, the agent's accumulated risk before this room
pAlive     = 1   -- probability the children in the room are alive
nEstimated = 2   -- estimated number of children
urgency    = 3   -- 1 (healthy), 3 (injured) or 5 (critical)
isClosed   = 4   -- 1 if nobody has seen inside the room, 0 if it is open

--------------------------------------------------------------------------------
-- Outputs

-- The network outputs a single number, the effort committed to the room.

type OutputVector = Tensor Real [1]

effortLevel = 0

--------------------------------------------------------------------------------
-- The network

-- The implementation is passed to the compiler as an ONNX file at compile time.
-- Unlike ACAS Xu, the network was trained on raw (unnormalised) inputs, so the
-- properties are written directly in the units of the data with no normalise step.

@network
effort : InputVector -> OutputVector

--------------------------------------------------------------------------------
-- Controller constants

-- Passed in at verification time from `data/meta.json`, in the same way the
-- drug-verification spec takes Ka, Ke, Vd and C_safe.

@parameter
tau : Real       -- score a room must exceed before any effort

@parameter
kEffort : Real   -- effort per unit of score above tau

@parameter
rMax : Real      -- risk at which the controller stops committing

-- How far below the controller's guaranteed effort the network may fall.
-- The network approximates the controller, so a strict bound (tolerance = 0)
-- can fail through approximation error alone.

@parameter
tolerance : Real

--------------------------------------------------------------------------------
-- Valid inputs

-- The ranges each input takes in the data.

minimumInputValues : InputVector
minimumInputValues = [0.0, 0.0, 1.0, 1.0, 0.0]

maximumInputValues : InputVector
maximumInputValues = [1.0, 1.0, 5.0, 5.0, 1.0]

inRange : InputVector -> Bool
inRange x = forall i . minimumInputValues ! i <= x ! i <= maximumInputValues ! i

-- A room is either open, in which case its children are known to be alive
-- (p_alive = 1), or closed. The urgency takes one of three levels.

validRoom : InputVector -> Bool
validRoom x =
  ((x ! isClosed == 0 and x ! pAlive == 1) or x ! isClosed == 1) and
  (x ! urgency == 1 or x ! urgency == 3 or x ! urgency == 5)

validInput : InputVector -> Bool
validInput x = inRange x and validRoom x

--------------------------------------------------------------------------------
-- P1: Maximise expected utility

-- The expected utility of a rescue is the number of children likely to be
-- alive, weighted by how urgently they need help:
--
--   EU = n_estimated * p_alive * urgency
--
-- A product of inputs cannot be stated directly to a linear verifier, so, like
-- the ACAS Xu regions, the room is described by lower bounds on each factor.
-- Together they guarantee a lower bound on its expected utility.

nLow = 4.0
pAliveLow = 0.8
urgencyLevel = 5.0

expectedUtilityLow : Real
expectedUtilityLow = nLow * pAliveLow * urgencyLevel   -- = 16

highExpectedUtility : InputVector -> Bool
highExpectedUtility x =
  x ! nEstimated >= nLow      and
  x ! pAlive     >= pAliveLow and
  x ! urgency    == urgencyLevel

-- The agent must also have risk budget left to act on it.

riskHigh = 0.2

lowRisk : InputVector -> Bool
lowRisk x = x ! risk <= riskHigh

-- The effort a room of this expected utility warrants: the controller's effort
-- at the least favourable point of the region (lowest EU, highest risk).

minimumEffort : Real
minimumEffort = kEffort * (expectedUtilityLow - tau) * (1 - riskHigh / rMax)   -- = 0.44

-- If a room's expected utility is certainly high and the agent is not at risk,
-- the agent commits at least the effort that expected utility warrants.
-- This holds for the controller by construction.

@property
maximiseExpectedUtility : Bool
maximiseExpectedUtility = forall x . validInput x and highExpectedUtility x and lowRisk x =>
  effort x ! effortLevel >= minimumEffort - tolerance
