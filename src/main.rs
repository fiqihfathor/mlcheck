use anyhow::Result;
use clap::{Parser, Subcommand};
use polars::prelude::*;

#[derive(Parser)]
#[command(name = "mlcheck")]
#[command(about = "Fast ML dataset validation CLI built in Rust - catch data issues before training", long_about=None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Inspect {
        file: String,
    },
    Validate {
        file: String,
        #[arg(short, long)]
        target: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { file } => {
            inspect_dataset(&file)?;
        }
        Commands::Validate { file, target } => {
            validate_dataset(&file, target.as_deref())?;
        }
    }

    Ok(())
}

fn read_csv(path: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()
}

fn inspect_dataset(path: &str) -> Result<()> {
    println!("🔍 Inspecting: {}\n", path);

    let df = read_csv(path)?;

    println!("📊 Dataset Overview");
    println!("├─ Rows: {}", df.height());
    println!("├─ Columns: {}", df.width());
    println!(
        "└─ Memory: {:.2} MB",
        df.estimated_size() as f64 / 1_000_000.0
    );

    println!("\n📋 Columns:");
    for col in df.get_columns() {
        println!("├─ {} ({})", col.name(), col.dtype());
    }

    Ok(())
}

fn validate_dataset(path: &str, target: Option<&str>) -> Result<()> {
    println!("✓ Validating: {}\n", path);

    let df = read_csv(path)?;

    // Basic Info
    println!("📊 Dataset Overview");
    println!("├─ Shape: {} rows × {} columns", df.height(), df.width());
    println!(
        "└─ Size: {:.2} MB\n",
        df.estimated_size() as f64 / 1_000_000.0
    );

    // Check missing values
    println!("🔍 Missing Values:");
    let mut has_missing = false;

    for col in df.get_columns() {
        let null_count = col.null_count();
        if null_count > 0 {
            has_missing = true;
            let percentage = (null_count as f64 / df.height() as f64) * 100.0;
            println!("├─ {}: {} ({:.1}%)", col.name(), null_count, percentage);
        }
    }

    if !has_missing {
        println!("└─ ✓ No missing values");
    }

    // Check duplicates
    println!("\n🔁 Duplicates:");

    let lf = df.clone().lazy();
    let deduped = lf.unique(None, UniqueKeepStrategy::First).collect()?;

    let duplicates = df.height() - deduped.height();

    if duplicates > 0 {
        println!(
            "└─ ⚠️  {} duplicate rows ({:.1}%)",
            duplicates,
            (duplicates as f64 / df.height() as f64) * 100.0
        );
    } else {
        println!("└─ ✓ No duplicates");
    }

    // Target column analysis
    if let Some(target_col) = target {
        println!("\n🎯 Target Column: {}", target_col);

        if let Ok(series) = df.column(target_col) {
            println!("├─ Type: {:?}", series.dtype());
            println!("├─ Unique values: {}", series.n_unique()?);

            let null_count = series.null_count();
            if null_count > 0 {
                println!(
                    "└─ ⚠️  Missing in target: {} ({:.1}%)",
                    null_count,
                    (null_count as f64 / df.height() as f64) * 100.0
                );
            } else {
                println!("└─ ✓ No missing values in target");
            }
        } else {
            println!("└─ ❌ Target column '{}' not found!", target_col);
        }
    }

    Ok(())
}
