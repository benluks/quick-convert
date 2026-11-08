def test_knnvc_hifigan_decoder_is_available_from_decoder_package():
    from quick_convert.components.generators.waveform.knnvc_hifigan import KnnVCHifiGanDecoder
    from quick_convert.components.generators.waveform.knnvc_hifigan import KnnVCHifiGanDecoder as DirectDecoder

    assert KnnVCHifiGanDecoder is DirectDecoder
