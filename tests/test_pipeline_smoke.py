# SPDX-License-Identifier: Apache-2.0


def test_main_exits_cleanly_when_image_folder_is_empty(tmp_path, monkeypatch, capsys):
    import main

    image_dir = tmp_path / "images"
    results_dir = tmp_path / "results"
    image_dir.mkdir()

    monkeypatch.setattr(main, "IMAGE_FOLDER", str(image_dir))
    monkeypatch.setattr(main, "RESULTS_FOLDER", str(results_dir))
    monkeypatch.setattr(main, "CSV_FILE", str(results_dir / "results.csv"))

    main.main()

    output = capsys.readouterr().out
    assert "No supported images found" in output
    assert not (results_dir / "results.csv").exists()


def test_main_ignores_unsupported_files(tmp_path, monkeypatch, capsys):
    import main

    image_dir = tmp_path / "images"
    results_dir = tmp_path / "results"
    image_dir.mkdir()
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    monkeypatch.setattr(main, "IMAGE_FOLDER", str(image_dir))
    monkeypatch.setattr(main, "RESULTS_FOLDER", str(results_dir))
    monkeypatch.setattr(main, "CSV_FILE", str(results_dir / "results.csv"))

    main.main()

    output = capsys.readouterr().out
    assert "No supported images found" in output
    assert not (results_dir / "results.csv").exists()


def test_main_returns_failure_status_when_image_processing_fails(tmp_path, monkeypatch, capsys):
    import main

    image_dir = tmp_path / "images"
    results_dir = tmp_path / "results"
    image_dir.mkdir()
    (image_dir / "sample.jpg").write_bytes(b"placeholder")

    monkeypatch.setattr(main, "IMAGE_FOLDER", str(image_dir))
    monkeypatch.setattr(main, "RESULTS_FOLDER", str(results_dir))
    monkeypatch.setattr(main, "CSV_FILE", str(results_dir / "results.csv"))
    monkeypatch.setattr(main, "process_image", lambda path: (_ for _ in ()).throw(RuntimeError("boom")))

    status = main.main()

    output = capsys.readouterr().out
    assert status == 1
    assert "Processing completed with 1 failure(s)." in output
    assert "Failed images: sample.jpg" in output
