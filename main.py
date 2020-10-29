from env import TargetNaiveEnv
from diffsim_agent import DiffsimAgent
from experiment import Experiment
import time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='default_out')
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--sim_time", type=float, default=0.01)

    args = parser.parse_args()
    out_path = "./exp/" + args.exp_name

    env = TargetNaiveEnv(sim_time=args.sim_time,
                         initial_pose=[0,0,0],
                         l = 1,
                         r = 0.5,
                         max_len=args.max_len)

    agent = DiffsimAgent(env.obs_dim, env.act_dim)
    exp = Experiment(env=env,agent=agent)

    for e in range(1000):
        # Rollout
        st = time.time()
        total_loss = exp.rollout_train()
        en0 = time.time()
        # update agent by diffsim gradient
        agent.update(total_loss)
            
        en1 = time.time()

        # visualize running time
        print("=====================================")
        print("epoch {}: loss = {}".format(
            e, total_loss.data
        ))
        print("foward time = {}".format(en0 - st))
        print("backward time = {}".format(en1 - en0))
