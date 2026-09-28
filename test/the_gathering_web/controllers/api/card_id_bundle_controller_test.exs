defmodule TheGatheringWeb.API.CardIdBundleControllerTest do
  use TheGatheringWeb.ConnCase, async: false

  alias TheGathering.CardId

  setup :register_and_log_in_user

  setup do
    root = CardId.bundle_dir()
    File.rm_rf!(root)
    on_exit(fn -> File.rm_rf!(root) end)
    %{root: root}
  end

  test "404 when nothing has been published", %{conn: conn} do
    assert %{"errors" => %{"detail" => "Not Found"}} =
             conn |> get(~p"/api/cardid/bundle") |> json_response(404)
  end

  test "describes the current bundle and serves its files immutably", %{conn: conn, root: root} do
    publish(root, "2026-09-22-full-3")

    assert %{"data" => data} = conn |> get(~p"/api/cardid/bundle") |> json_response(200)

    assert %{
             "version" => "2026-09-22-full-3",
             "gallery" => %{"arts" => 2, "topk" => 5},
             "constants" => %{"scene" => 640},
             "files" => %{"detector.onnx" => detector_url, "arts.json" => arts_url}
           } = data

    assert detector_url == "/api/cardid/bundles/2026-09-22-full-3/detector.onnx"

    response = get(conn, detector_url)
    assert response.status == 200
    assert response.resp_body == "onnx-bytes"
    assert get_resp_header(response, "content-type") == ["application/octet-stream"]
    assert get_resp_header(response, "cache-control") == ["private, max-age=31536000, immutable"]

    assert [%{"name" => "Forest"}, _] = conn |> get(arts_url) |> json_response(200)
  end

  test "an old version stays addressable after current moves on", %{conn: conn, root: root} do
    publish(root, "v1")
    publish(root, "v2")

    assert %{"data" => %{"version" => "v2"}} =
             conn |> get(~p"/api/cardid/bundle") |> json_response(200)

    assert conn |> get(~p"/api/cardid/bundles/v1/embed.onnx") |> response(200)
  end

  test "only advertises the optional sibling file once it exists on disk", %{
    conn: conn,
    root: root
  } do
    publish(root, "v1")
    data = conn |> get(~p"/api/cardid/bundle") |> json_response(200)
    refute Map.has_key?(data["data"]["files"], "printings.json")

    manifest_path = Path.join([root, "v1", "manifest.json"])
    manifest = manifest_path |> File.read!() |> Jason.decode!()
    File.write!(manifest_path, Jason.encode!(put_in(manifest, ["files", "printings.json"], %{})))
    File.write!(Path.join([root, "v1", "printings.json"]), ~s({"a":[{"id":"sibling"}]}))
    data = conn |> get(~p"/api/cardid/bundle") |> json_response(200)
    url = data["data"]["files"]["printings.json"]
    assert url == "/api/cardid/bundles/v1/printings.json"
    response = get(conn, url)
    assert json_response(response, 200) == %{"a" => [%{"id" => "sibling"}]}
    assert get_resp_header(response, "cache-control") == ["private, max-age=31536000, immutable"]
  end

  test "advertises the table detector once published, and serves a detection-only bundle", %{
    conn: conn,
    root: root
  } do
    publish(root, "v1")
    data = conn |> get(~p"/api/cardid/bundle") |> json_response(200)
    refute Map.has_key?(data["data"]["files"], "table_detector.onnx")

    manifest_path = Path.join([root, "v1", "manifest.json"])
    manifest = manifest_path |> File.read!() |> Jason.decode!()
    File.write!(manifest_path, Jason.encode!(put_in(manifest, ["files", "table_detector.onnx"], %{})))
    File.write!(Path.join([root, "v1", "table_detector.onnx"]), "table-onnx-bytes")
    data = conn |> get(~p"/api/cardid/bundle") |> json_response(200)
    url = data["data"]["files"]["table_detector.onnx"]
    assert url == "/api/cardid/bundles/v1/table_detector.onnx"
    response = get(conn, url)
    assert response.resp_body == "table-onnx-bytes"
    assert get_resp_header(response, "content-type") == ["application/octet-stream"]

    # No embedding pipeline published yet: gallery/constants/arts.json are absent entirely.
    detection_only = %{
      "version" => "detector-only",
      "created" => "2026-09-26T00:00:00+00:00",
      "files" => %{"table_detector.onnx" => %{}}
    }

    dir = Path.join(root, "detector-only")
    File.mkdir_p!(dir)
    File.write!(Path.join(dir, "manifest.json"), Jason.encode!(detection_only))
    File.write!(Path.join(dir, "table_detector.onnx"), "table-onnx-bytes")
    current = Path.join(root, "current")
    File.rm(current)
    File.ln_s!("detector-only", current)

    assert %{"data" => data} = conn |> get(~p"/api/cardid/bundle") |> json_response(200)
    assert data["version"] == "detector-only"
    assert data["gallery"] == nil
    assert data["constants"] == nil
    assert data["files"] == %{
             "manifest.json" => "/api/cardid/bundles/detector-only/manifest.json",
             "table_detector.onnx" => "/api/cardid/bundles/detector-only/table_detector.onnx"
           }
  end

  test "refuses files outside the bundle", %{conn: conn, root: root} do
    publish(root, "v1")
    File.write!(Path.join(root, "secret.txt"), "nope")

    for path <- [
          "/api/cardid/bundles/v1/SHA256SUMS",
          "/api/cardid/bundles/current/manifest.json",
          "/api/cardid/bundles/missing/manifest.json",
          "/api/cardid/bundles/..%2F/secret.txt",
          "/api/cardid/bundles/v1/..%2Fsecret.txt"
        ] do
      assert conn |> get(path) |> json_response(404), path
    end
  end

  test "requires authentication", %{conn: conn, root: root} do
    publish(root, "v1")
    conn = conn |> recycle() |> init_test_session(%{})
    previous = Application.get_env(:the_gathering, :dev_auto_login, false)
    Application.put_env(:the_gathering, :dev_auto_login, false)
    on_exit(fn -> Application.put_env(:the_gathering, :dev_auto_login, previous) end)

    assert conn |> get(~p"/api/cardid/bundle") |> json_response(401)
    assert conn |> get(~p"/api/cardid/bundles/v1/detector.onnx") |> json_response(401)
  end

  # Lays out a bundle the way `python -m cardid.publish` does and points `current` at it.
  defp publish(root, version) do
    dir = Path.join(root, version)
    File.mkdir_p!(dir)

    manifest = %{
      "version" => version,
      "created" => "2026-09-22T00:00:00+00:00",
      "gallery" => %{
        "arts" => 2,
        "dtype" => "f16",
        "embed_dim" => 128,
        "frame_penalty" => 0.02,
        "topk" => 5
      },
      "constants" => %{"scene" => 640, "det_input" => 256},
      "files" => %{}
    }

    File.write!(Path.join(dir, "manifest.json"), Jason.encode!(manifest))

    File.write!(
      Path.join(dir, "arts.json"),
      Jason.encode!([
        %{"id" => "a", "name" => "Forest", "set" => "fin"},
        %{"id" => "b", "name" => "Island", "set" => "fin"}
      ])
    )

    for name <- ~w(detector.onnx embed.onnx search.onnx),
        do: File.write!(Path.join(dir, name), "onnx-bytes")

    File.write!(Path.join(dir, "SHA256SUMS"), "sums")

    current = Path.join(root, "current")
    File.rm(current)
    File.ln_s!(version, current)
  end
end
