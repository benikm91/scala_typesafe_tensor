ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.6.4"

lazy val root = project
  .in(file("."))
  .settings(
    name := "shape-safety-tensor",
    resolvers ++= Resolver.sonatypeOssRepos("snapshots"),
    libraryDependencies ++= Seq(
      "dev.storch" %% "core" % "0.0-2dfa388-SNAPSHOT",
      "org.bytedeco" % "pytorch" % "2.1.2-1.5.10",
      "org.bytedeco" % "pytorch" % "2.1.2-1.5.10" classifier "macosx-arm64",
      "org.bytedeco" % "openblas" % "0.3.26-1.5.10" classifier "macosx-arm64",
    ),
    fork := true
  )
