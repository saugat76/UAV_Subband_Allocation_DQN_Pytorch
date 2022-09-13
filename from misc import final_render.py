from misc import final_render

state = [[2, 1], [3, 6], [7, 4], [7, 8], [8, 1]]
state.reshape(1, 5 * 3)
remark = "best"
final_render(state, "best")