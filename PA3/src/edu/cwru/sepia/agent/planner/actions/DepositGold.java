package edu.cwru.sepia.agent.planner.actions;

import edu.cwru.sepia.agent.planner.GameState;
import edu.cwru.sepia.environment.model.state.ResourceType;

public class DepositGold implements StripsAction {
	
	private GameState.Peasant peasant;
	private GameState.TownHall townHall;
	
	public DepositGold(GameState.Peasant peasant, GameState.TownHall townHall) {
		this.peasant = peasant;
		this.townHall = townHall;
	}
	
	public boolean preconditionsMet(GameState state) {
		GameState.Peasant statePeasant = state.getPeasant();
		GameState.TownHall stateTownHall = state.getTownHall();
		
		if (peasant.getPosition().equals(statePeasant.getPosition()) && townHall.getPosition().equals(stateTownHall.getPosition())) {
			return peasant.getPosition().isAdjacent(townHall.getPosition()) &&
					peasant.getCargoType() == ResourceType.GOLD &&
					peasant.getCargoAmount() > 0;
		} else {
			return false;
		}
	}

	public GameState apply(GameState gameState) {
		GameState result = new GameState(gameState, this);
		
		GameState.Peasant resultPeasant = result.getPeasant();
		result.addGold(resultPeasant.getCargoAmount());
		resultPeasant.deposit();
		
		return result;
	}
}
