from ql.maze_env import Maze

from ql.RL_brain import QLearningTable
from ql.RL_brain import SarsaTable
from ql.RL_brain import SarsaLambdaTable


def update():
    # 学习 100 回合
    for episode in range(100):
        # 初始化 state 的观测值
        observation = env.reset()
        RL.eligibility_trace *= 0
        action = RL.choose_action(str(observation))
        while True:
            # 更新可视化环境
            env.render()

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done = env.step(action)
            # RL 大脑根据 state 的观测值挑选 action
            action_ = RL.choose_action(str(observation_))

            # RL 从这个序列 (state, action, reward, state_) 中学习
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个 state 的值传到下一次循环
            observation = observation_
            action = action_

            # 如果掉下地狱或者升上天堂, 这回合就结束了
            if done:
                break

    # 结束游戏并关闭窗口
    print('game over')

    print('\r\nQ-table:\n')
    print(RL.q_table)
    env.destroy()


if __name__ == "__main__":
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()
