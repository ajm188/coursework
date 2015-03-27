package edu.cwru.sepia.agent.planner.actions;

import java.io.IOException;
import java.util.List;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.agent.planner.Position;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class DepositGold implements StripsAction {

	private Position peasantPos;
	private Position townHallPos;
	private Position targetPosition;
	
	public DepositGold(Position peasantPos, Position townHallPos){
		this.peasantPos = peasantPos;
		this.townHallPos = townHallPos;		
	}
	
	public boolean preconditionsMet(GameState state) {
		Unit.UnitView peasantView = state.getPeasantView();
		Unit.UnitView townHallView = state.getTownHallView();

		if (peasantView == null || townHallView == null) {
			return false;
		}
		
		Position peasantPosition = new Position(peasantView.getXPosition(), peasantView.getYPosition());
		Position townHallPosition = new Position(townHallView.getXPosition(), townHallView.getYPosition());
		
		List<Position> townHallAdjacents = townHallPosition.getAdjacentPositions();
		boolean adjEmpty = false;
		for (Position adjacentPosition : townHallAdjacents) {
			if (!(state.getStateView().isResourceAt(adjacentPosition.x, adjacentPosition.y) || state.getStateView().isUnitAt(adjacentPosition.x, adjacentPosition.y))) {
				this.targetPosition = adjacentPosition;
				adjEmpty = true;
				break;
			}
		}
		
		return peasantPos.equals(peasantPosition) &&
				townHallPos.equals(townHallPosition) &&
				adjEmpty &&
				peasantView.getCargoAmount() == 0 &&
				peasantView.getCargoType() == ResourceType.GOLD;
	}

	public GameState apply(GameState gameState) {
		State state;
		try {
			state = gameState.getStateView().getStateCreator().createState();
		} catch (IOException e) {
			return null;
		}
		
		Unit peasant = state.getUnit(gameState.getStateView().unitAt(peasantPos.x, peasantPos.y));
		
		state.transportUnit(peasant, targetPosition.x, targetPosition.y); // move the peasant next to the townhall
		peasant.setCargo(ResourceType.GOLD, 0);
		state.addResourceAmount(gameState.getPlayerNum(), ResourceType.GOLD, 100);
		
		return new GameState(state.getView(gameState.getPlayerNum()),
				gameState.getPlayerNum(),
				gameState.getRequiredGold(),
				gameState.getRequiredWood(),
				gameState.getBuildPeasants(),
				gameState,
				this);	
	}
	
}
