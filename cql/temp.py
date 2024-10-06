import gymnasium as gym
import mujoco
import imageio

# 환경을 만들 때 render_mode를 'rgb_array'로 설정
env = gym.make("HalfCheetah-v4", render_mode='rgb_array')

video_filename = 'halfcheetah_run.mp4'
video_writer = imageio.get_writer(video_filename, fps=30)

# 환경 초기화
state = env.reset()
done = False
frames = []

cnt = 0
while not done:
    action = env.action_space.sample()  # 무작위 액션 선택
    state, reward, done, truncated, info = env.step(action)
    done = done or truncated
    
    # 렌더링된 프레임 가져오기
    frame = env.render()  # 더 이상 mode='rgb_array'가 필요하지 않습니다.
    frames.append(frame)

    if cnt % 100 == 0:
        print(cnt)
    cnt += 1

for frame in frames:
    video_writer.append_data(frame)

video_writer.close()

env.close()