defmodule TheGatheringWeb.API.CardIdBundleController do
  @moduledoc """
  Serves the published card-recognition bundle (`TheGathering.CardId`) to the webcam table.

  `show` describes the current bundle and where its files live; `file` streams one file of a
  specific version. Files are addressed by version so the browser can cache them forever and
  `show` alone decides when a newer version should be fetched.
  """

  use TheGatheringWeb, :controller

  alias TheGathering.CardId

  action_fallback TheGatheringWeb.API.FallbackController

  @content_types %{
    "manifest.json" => "application/json",
    "arts.json" => "application/json",
    "printings.json" => "application/json",
    "detector.onnx" => "application/octet-stream",
    "embed.onnx" => "application/octet-stream",
    "search.onnx" => "application/octet-stream",
    "table_detector.onnx" => "application/octet-stream"
  }

  def show(conn, _params) do
    with {:ok, manifest} <- CardId.current_manifest() do
      # Only files this version actually has on disk (not just named in the manifest's own
      # `files` key): the table detector may ship years before the embedding pipeline
      # (arts.json/detector.onnx/embed.onnx/search.onnx) exists, or the other way around, and
      # printings.json is always sibling-file optional.
      files =
        CardId.files()
        |> Enum.filter(&match?({:ok, _}, CardId.file_path(manifest["version"], &1)))
        |> Map.new(fn name ->
          {name, ~p"/api/cardid/bundles/#{manifest["version"]}/#{name}"}
        end)

      conn
      |> put_resp_header("cache-control", "private, no-cache")
      |> render(:show, manifest: manifest, files: files)
    end
  end

  def file(conn, %{"version" => version, "name" => name}) do
    with {:ok, path} <- CardId.file_path(version, name) do
      conn
      |> put_resp_content_type(Map.fetch!(@content_types, name), nil)
      |> put_resp_header("cache-control", "private, max-age=31536000, immutable")
      |> send_file(200, path)
    end
  end
end
