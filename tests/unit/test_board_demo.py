from classroom_ai.video.board_demo import BoardDemoState


def test_offscreen_destination_never_claims_visible_writing():
    s = BoardDemoState([0, 0.8, 1, 1], offscreen=True)
    for i, y in enumerate((60, 70, 80, 90)):
        hints = s.update(
            [dict(track_id=1, state="observed", confirmed=True, bbox_xyxy=[45, y - 20, 55, y])],
            segment=0,
            t=i * 0.1,
            width=100,
            height=100,
        )[1]
    assert hints[0]["status"] == "candidate"
    assert hints[1]["status"] == "no_suggestion"


def test_user_demo_near_board_is_writing_and_approaching_is_moving():
    s = BoardDemoState([0.4, 0.2, 0.6, 0.4])

    def row(y):
        return dict(track_id=1, state="observed", confirmed=True, bbox_xyxy=[45, y - 10, 55, y])

    assert (
        s.update([row(80)], segment=0, t=0, width=100, height=100)[1][0]["status"]
        == "no_suggestion"
    )
    s.update([row(75)], segment=0, t=0.1, width=100, height=100)
    s.update([row(70)], segment=0, t=0.2, width=100, height=100)
    hints = s.update([row(65)], segment=0, t=0.3, width=100, height=100)[1]
    assert hints[0]["status"] == "candidate"
    hints = s.update([row(35)], segment=0, t=0.4, width=100, height=100)[1]
    assert hints[1]["status"] == "active"
    assert hints[1]["evidence_scope"] == "user_demo_proximity_not_visual_writing"
    s.update([], segment=0, t=1.1, width=100, height=100)
    assert (
        s.update([row(65)], segment=0, t=1.2, width=100, height=100)[1][0]["status"]
        == "no_suggestion"
    )
