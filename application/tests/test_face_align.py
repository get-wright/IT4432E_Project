from application.backend.face_align import FaceAligner


def test_aligner_records_size_and_norm():
    a = FaceAligner(input_size=112, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert a.input_size == 112
    # MTCNN must be configured at the requested size.
    assert a.mtcnn.image_size == 112
