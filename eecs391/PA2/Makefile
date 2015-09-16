JAVAC=javac
SEPIA_JAR=../jars/Sepia.jar
SEPIA_MAIN=edu.cwru.sepia.Main2
SEPIA_CP=-cp $(SEPIA_JAR)
SRC=src
ARCHER_CP=archer_agent
JAVA_CMD=java $(SEPIA_CP):$(SRC):$(ARCHER_CP) $(SEPIA_MAIN)
AGENT_PATH=$(SRC)/edu/cwru/sepia/agent/minimax/
OUTPUT=$(shell ls $(AGENT_PATH)*.java)


all: run

run: agent
	$(JAVA_CMD) data/GameConfig2fv2a.xml

%.xml: agent
	$(JAVA_CMD) data/$@
	
agent: $(OUTPUT)
	$(JAVAC) $(SEPIA_CP) $^

clean:
	rm -f $(shell ls $(AGENT_PATH)*.class)

