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
	
	public GameState.Peasant getPeasant() {
		return this.peasant;
	}
	
	public GameState.TownHall getTownHall() {
		return this.townHall;
	}
	
	public boolean preconditionsMet(GameState gameState) {
		GameState.Peasant peasant = gameState.getPeasant();
		GameState.TownHall townHall = gameState.getTownHall();
		
		if (peasant.getID() != this.peasant.getID() || townHall.getID() != this.townHall.getID()) {
			return false;
		}
		
		return peasant.getCargoType() == this.peasant.getCargoType() &&
				peasant.getCargoType() == ResourceType.GOLD &&
				peasant.getCargoAmount() == this.peasant.getCargoAmount() &&
				peasant.getCargoAmount() > 0;
	}

	public GameState apply(GameState gameState) {
		GameState result = new GameState(gameState, this);
		
		GameState.Peasant resultPeasant = result.getPeasant();
		result.addGold(resultPeasant.getCargoAmount());
		resultPeasant.deposit();
		
		return result;
	}
}
