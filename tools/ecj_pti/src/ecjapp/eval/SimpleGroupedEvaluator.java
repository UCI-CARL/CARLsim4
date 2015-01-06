package ecjapp.eval;

import ecjapp.eval.problem.SimpleGroupedProblemForm;
import ec.Evaluator;
import ec.EvolutionState;
import ec.Fitness;
import ec.Individual;
import ec.Population;
import ec.Subpopulation;
import ec.util.Parameter;
import ec.util.ThreadPool;
import java.io.File;
import java.io.IOException;

/**
 * A simple evaluator for cases in which individuals must be evaluated as
 * groups in a single external call (such as to a GPU simulation library). The
 * evaluation is not necessarily coevolutionary, but neither can individuals
 * be evaluated one at a time (as assumed by SimpleEvaluator).
 * 
 * @author Eric 'Siggy' Scott
 * 
 * Based on Sean Luke's SimpleEvaluator.
 */
public class SimpleGroupedEvaluator extends Evaluator {
    public static final String P_CLONE_PROBLEM = "clone-problem";
    
    public static final String P_NUM_TESTS = "num-tests";
    public static final String P_MERGE = "merge";
    public static final String V_MEAN = "mean";
    public static final String V_MEDIAN = "median";
    public static final String V_BEST = "best";
        
    public static final String P_CHUNK_SIZE = "chunk-size";
    public static final String V_AUTO = "auto";
    
    public static final String P_MEASURE_EVALTIME = "measureEvalTimes";
    public static final String P_EVALTIME_FILE = "evalTimesFile";

    public static final int MERGE_MEAN = 0;
    public static final int MERGE_MEDIAN = 1;
    public static final int MERGE_BEST = 2;

    public int numTests = 1;
    public int mergeForm = MERGE_MEAN;
    public boolean cloneProblem;

    final Object[] lock = new Object[0];          // Arrays are serializable
    int individualCounter = 0;
    int subPopCounter = 0;
    int chunkSize;  // a value >= 1, or C_AUTO
    public static final int C_AUTO = 0;
        
    public ThreadPool pool = new ThreadPool();
    
    private boolean measureEvalTimes;
    private int evalTimeLog = 0;  // default to stdout
    private long startTime;

    // checks to make sure that the Problem implements SimpleGroupedProblemForm
    @Override
    public void setup(final EvolutionState state, final Parameter base) {
        super.setup(state,base);
        if (!(p_problem instanceof SimpleGroupedProblemForm))
            state.output.fatal("" + this.getClass().getSimpleName() + " used, but the Problem is not of SimpleGroupedProblemForm",
                base.push(P_PROBLEM));

        cloneProblem =state.parameters.getBoolean(base.push(P_CLONE_PROBLEM), null, true);
        if (!cloneProblem && (state.breedthreads > 1)) // uh oh, this can't be right
            state.output.fatal("The Evaluator is not cloning its Problem, but you have more than one thread.", base.push(P_CLONE_PROBLEM));

        numTests = state.parameters.getInt(base.push(P_NUM_TESTS), null, 1);
        if (numTests < 1) numTests = 1;
        else if (numTests > 1) {
            String m = state.parameters.getString(base.push(P_MERGE), null);
            if (m == null)
                state.output.warning("Merge method not provided to SimpleEvaluator.  Assuming 'mean'");
            else if (m.equals(V_MEAN))
                mergeForm = MERGE_MEAN;
            else if (m.equals(V_MEDIAN))
                mergeForm = MERGE_MEDIAN;
            else if (m.equals(V_BEST))
                mergeForm = MERGE_BEST;
            else
                state.output.fatal("Bad merge method: " + m, base.push(P_NUM_TESTS), null);
        }
                
        if (!state.parameters.exists(base.push(P_CHUNK_SIZE), null)) {
            chunkSize = C_AUTO;
        }
        else if (state.parameters.getString(base.push(P_CHUNK_SIZE), null).equalsIgnoreCase(V_AUTO)) {
            chunkSize = C_AUTO;
        }
        else {
            chunkSize = (state.parameters.getInt(base.push(P_CHUNK_SIZE), null, 1));
            if (chunkSize == 0)  // uh oh
                state.output.fatal("Chunk Size must be either an integer >= 1 or 'auto'", base.push(P_CHUNK_SIZE), null);
        }
        
        measureEvalTimes = state.parameters.getBoolean(base.push(P_MEASURE_EVALTIME), null, false);
        if (state.parameters.exists(base.push(P_EVALTIME_FILE), null)) {
            if (measureEvalTimes = false)
                state.output.warnOnce(String.format("Parameter %s is being ignored because %s is set to false.", base.push(P_EVALTIME_FILE), base.push(P_MEASURE_EVALTIME)));
            measureEvalTimes = true;
            final File evalTimeFile = state.parameters.getFile(base.push(P_EVALTIME_FILE), null);
            try {
                evalTimeLog = state.output.addLog(evalTimeFile, false, true);
            } catch (final IOException i) {
                state.output.fatal("An IOException occurred while trying to create the log " + evalTimeFile + ":\n" + i);
            }
        }
        
        startTime = System.currentTimeMillis();
    } 

    Population oldpop = null;
    // replace the population with one that has some N copies of the original individuals
    void expand(EvolutionState state)
        {
        Population pop = (Population)(state.population.emptyClone());

        // populate with clones
        for(int i = 0; i < pop.subpops.length; i++)
            {
            pop.subpops[i].individuals = new Individual[numTests * state.population.subpops[i].individuals.length];
            for(int j = 0; j < state.population.subpops[i].individuals.length; j++)
                {
                for (int k=0; k < numTests; k++)
                    {
                    pop.subpops[i].individuals[numTests * j + k] =
                        (Individual)(state.population.subpops[i].individuals[j].clone());
                    }
                }
            }

        // swap
        oldpop = state.population; // FIXME
        state.population = pop;
        }

    // Take the N copies of the original individuals and fold their fitnesses back into the original
    // individuals, replacing them with the original individuals in the process.  See expand(...)
    void contract(EvolutionState state)
        {
        // swap back
        Population pop = state.population;
        state.population = oldpop;

        // merge fitnesses again
        for(int i = 0; i < pop.subpops.length; i++)
            {
            Fitness[] fits = new Fitness[numTests];
            for(int j = 0; j < state.population.subpops[i].individuals.length; j++)
                {
                for (int k=0; k < numTests; k++)
                    {
                    fits[k] = pop.subpops[i].individuals[numTests * j + k].fitness;
                    }

                if (mergeForm == MERGE_MEAN)
                    {
                    state.population.subpops[i].individuals[j].fitness.setToMeanOf(state, fits);
                    }
                else if (mergeForm == MERGE_MEDIAN)
                    {
                    state.population.subpops[i].individuals[j].fitness.setToMedianOf(state, fits);
                    }
                else  // MERGE_BEST
                    {
                    state.population.subpops[i].individuals[j].fitness.setToBestOf(state, fits);
                    }

                state.population.subpops[i].individuals[j].evaluated = true;
                }
            }
        }
        
    /** A simple evaluator that doesn't do any coevolutionary
        evaluation.  Basically it applies evaluation pipelines,
        one per thread, to various subchunks of a new population. */
    @Override
    public void evaluatePopulation(final EvolutionState state) {
        assert(state.evalthreads > 0);
        
        if (measureEvalTimes)
            state.output.print(String.format("%d,%d,%d,", state.job[0], state.generation, System.currentTimeMillis() - startTime), evalTimeLog);
        
        if (numTests > 1)
            expand(state);
            
        // reset counters.  Only used in multithreading
        individualCounter = 0;
        subPopCounter = 0;

        ThreadPool.Worker[] threads = new ThreadPool.Worker[state.evalthreads];
        for(int i = 0; i < threads.length; i++)
        {
            SimpleEvaluatorThread run = new SimpleEvaluatorThread();
            run.threadnum = i;
            run.state = state;
            run.prob = (SimpleGroupedProblemForm)(p_problem.clone());
            threads[i] = pool.start(run, "ECJ Evaluation Thread " + i);
        }

        // join
        pool.joinAll();

        if (numTests > 1)
            contract(state);
        
        if (measureEvalTimes)
            state.output.println(String.format("%d", System.currentTimeMillis() - startTime), evalTimeLog);
    }


    /** The SimpleEvaluator determines that a run is complete by asking
        each individual in each population if he's optimal; if he 
        finds an individual somewhere that's optimal,
        he signals that the run is complete. */
    @Override
    public boolean runComplete(final EvolutionState state)
        {
        for(int x = 0;x<state.population.subpops.length;x++)
            for(int y=0;y<state.population.subpops[x].individuals.length;y++)
                if (state.population.subpops[x].
                    individuals[y].fitness.isIdealFitness())
                    return true;
        return false;
        }



    /** A private helper function for evaluatePopulation which evaluates a chunk
        of individuals in a subpopulation for a given thread.
        Although this method is declared
        protected, you should not call it. */
    private void evalPopChunk(final EvolutionState state, final int[] numinds, final int[] from,
        final int threadnum, final SimpleGroupedProblemForm p)
        {
        ((ec.Problem)p).prepareToEvaluate(state,threadnum);
        
        Subpopulation[] subpops = state.population.subpops;
        int len = subpops.length;
        
        for(int pop=0;pop<len;pop++) {
            // start evaluatin'!
            int fp = from[pop];
            int upperbound = fp+numinds[pop];
            Individual[] inds = subpops[pop].individuals;
            p.evaluate(state, inds, fp, upperbound, pop, threadnum);
        }
                        
        ((ec.Problem)p).finishEvaluating(state,threadnum);
        }


    // computes the chunk size if 'auto' is set.  This may be different depending on the subpopulation,
    // which is backward-compatible with previous ECJ approaches.
    private int computeChunkSizeForSubpopulation(EvolutionState state, int subpop, int threadnum)
        {
        int numThreads = state.evalthreads;
                
        // we will have some extra individuals.  We distribute these among the early subpopulations
        int individualsPerThread = state.population.subpops[subpop].individuals.length / numThreads;  // integer division
        int slop = state.population.subpops[subpop].individuals.length - numThreads * individualsPerThread;
                
        if (threadnum >= slop) // beyond the slop
            return individualsPerThread;
        else return individualsPerThread + 1;
        }



    /** A helper class for implementing multithreaded evaluation */
    private class SimpleEvaluatorThread implements Runnable
        {
        public int threadnum;
        public EvolutionState state;
        public SimpleGroupedProblemForm prob = null;
        
        @Override
        public void run() 
            {
            Subpopulation[] subpops = state.population.subpops;

            int[] numinds = new int[subpops.length];
            int[] from = new int[subpops.length];

            int count = 1;
            int start = 0;
            int subpop = 0;

            while (true)
                {
                // We need to grab the information about the next chunk we're responsible for.  This stays longer
                // in the lock than I'd like :-(
                synchronized(lock)
                    {
                    // has everyone done all the jobs?
                    if (subPopCounter >= subpops.length) // all done
                        return;
                                        
                    // has everyone finished the jobs for this subpopulation?
                    if (individualCounter >= subpops[subPopCounter].individuals.length)  // try again, jump to next subpop
                        {
                        individualCounter = 0; 
                        subPopCounter++;
                                        
                        // has everyone done all the jobs?  Check again.
                        if (subPopCounter >= subpops.length) // all done
                            return;
                        }

                    start = individualCounter;
                    subpop = subPopCounter;
                    count = chunkSize;
                    if (count == C_AUTO)  // compute automatically for subpopulations
                        count = computeChunkSizeForSubpopulation(state, subpop, threadnum);
                    
                    individualCounter += count;  // it can be way more than we'll actually do, that's fine                    
                    }
                
                // Modify the true count
                if (count >= subpops[subpop].individuals.length - start)
                    count = subpops[subpop].individuals.length - start;

                // Load into arrays to reuse evalPopChunk
                for(int i = 0; i < from.length; i++)
                    numinds[i] = 0;
                        
                numinds[subpop] = count;
                from[subpop] = start;
                evalPopChunk(state, numinds, from, threadnum, prob);
                }
            }
        }


    }

